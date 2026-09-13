use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

use anyhow::{anyhow, bail, Context, Result};
use lewton::audio::{read_audio_packet_generic, PreviousWindowRight};
use lewton::header::{
    read_header_comment, read_header_ident, read_header_setup, CommentHeader, SetupHeader,
};
use lewton::samples::InterleavedSamples;
use ogg::reading::PacketReader;
use theora_rs::{HeaderParser, OggPacket, PixelFmt};

pub const TH_PF_420: i32 = 0;
pub const TH_PF_422: i32 = 2;
pub const TH_PF_444: i32 = 3;

#[derive(Debug, Clone, Copy)]
pub struct VideoInfo {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub fmt: i32,
}

#[derive(Debug, Clone)]
struct OggPacketMeta {
    data: Vec<u8>,
    b_o_s: bool,
    e_o_s: bool,
    granulepos: i64,
    packetno: i64,
}

#[derive(Debug, Clone, Default)]
struct LogicalStream {
    packets: Vec<OggPacketMeta>,
}

pub struct TheoraFile {
    info: VideoInfo,
    video_frames: Vec<Vec<u8>>,
    video_cursor: usize,
    audio_channels: i32,
    audio_sample_rate: i32,
    has_audio_stream: bool,
    audio_samples: Vec<f32>,
    audio_cursor: usize,
}

/// Incremental Theora video decoder over an Ogg reader.
///
/// This keeps the Ogg packet reader and Theora decoder alive and decodes only
/// the next requested video packet. It deliberately ignores non-Theora logical
/// streams, so an OMV playback worker does not have to read the complete file or
/// decode its full video before the first frame can be displayed.
pub struct TheoraVideoStream<R: Read + Seek> {
    reader: PacketReader<R>,
    video_serial: u32,
    decoder: theora_rs::DecoderContext,
    raw_info: theora_rs::Info,
    info: VideoInfo,
    /// Number of Theora header packets consumed before the first video packet.
    header_packet_count: i64,
    packet_no: i64,
}

impl<R: Read + Seek> TheoraVideoStream<R> {
    pub fn open(reader: R) -> Result<Self> {
        let mut reader = PacketReader::new(reader);
        let mut parser = HeaderParser::new();
        let mut video_serial = None;
        let mut packet_no = 0i64;

        while let Some(pkt) = reader.read_packet().context("ogg packet read")? {
            let serial = pkt.stream_serial();
            if video_serial.is_none()
                && pkt.first_in_stream()
                && pkt.data.len() >= 7
                && pkt.data[0] == 0x80
                && &pkt.data[1..7] == b"theora"
            {
                video_serial = Some(serial);
            }
            if Some(serial) != video_serial {
                continue;
            }

            let b_o_s = pkt.first_in_stream();
            let e_o_s = pkt.last_in_stream();
            let granulepos = pkt.absgp_page() as i64;
            let data = pkt.data;
            let op = OggPacket {
                packet: data,
                b_o_s,
                e_o_s,
                granulepos,
                packetno: packet_no,
            };
            packet_no = packet_no.saturating_add(1);
            let _ = parser.push(&op)?;
            if parser.is_ready() {
                let decoder = parser.decoder()?;
                let raw_info = parser.info.clone();
                let info = video_info_from_theora_info(&raw_info);
                return Ok(Self {
                    reader,
                    video_serial: serial,
                    decoder,
                    raw_info,
                    info,
                    header_packet_count: packet_no,
                    packet_no,
                });
            }
        }

        bail!("stream ended before all Theora headers were parsed")
    }

    pub fn info(&self) -> VideoInfo {
        self.info
    }

    pub fn read_video_frame(&mut self) -> Result<Option<Vec<u8>>> {
        while let Some(pkt) = self.reader.read_packet().context("ogg packet read")? {
            if pkt.stream_serial() != self.video_serial {
                continue;
            }
            let b_o_s = pkt.first_in_stream();
            let e_o_s = pkt.last_in_stream();
            let granulepos = pkt.absgp_page() as i64;
            let data = pkt.data;
            let op = OggPacket {
                packet: data,
                b_o_s,
                e_o_s,
                granulepos,
                packetno: self.packet_no,
            };
            self.packet_no = self.packet_no.saturating_add(1);
            // Empty Ogg packets advance granule state but do not produce a new
            // decoded frame. Reset this edge-triggered flag before each packet
            // so the previous frame is never emitted twice.
            self.decoder.frame_available = false;
            theora_rs::th_decode_packetin(&mut self.decoder, &op)?;
            if self.decoder.has_decoded_frame() {
                let ycbcr = theora_rs::th_decode_ycbcr_out(&self.decoder)?;
                return Ok(Some(pack_theorafile_frame(&self.raw_info, &ycbcr)?));
            }
        }
        Ok(None)
    }

    /// Seek to an OMV packet using the original page/key-frame index.
    ///
    /// `file_offset` points to the indexed `seek_page_no`, which can precede
    /// the key-frame page when an Ogg packet spans pages. `first_packet_no` is
    /// the key-frame page's `top_packet_no`: this is where tona3 starts its
    /// decode packet counter after feeding/draining the back-pages.  In
    /// particular, the seek page itself is allowed to have `top_packet_no=-1`.
    /// Packets completed before `key_page_file_offset` must be drained without
    /// advancing that counter. The retained decoder context decodes the key frame through
    /// `target_packet_no`.
    pub fn seek_to_indexed_frame(
        &mut self,
        file_offset: u64,
        key_page_file_offset: u64,
        first_packet_no: usize,
        key_frame_packet_no: usize,
        target_packet_no: usize,
    ) -> Result<Option<Vec<u8>>> {
        if key_page_file_offset < file_offset {
            bail!("Theora key-frame page precedes seek page");
        }
        if first_packet_no > key_frame_packet_no {
            bail!(
                "Theora seek packet {} follows key packet {}",
                first_packet_no,
                key_frame_packet_no
            );
        }
        if target_packet_no < key_frame_packet_no {
            bail!(
                "Theora target packet {} precedes key packet {}",
                target_packet_no,
                key_frame_packet_no
            );
        }

        self.reader
            .seek_bytes(SeekFrom::Start(file_offset))
            .context("seek indexed Ogg page")?;

        let mut data_packet_no = first_packet_no;
        let mut reached_key_page = false;
        while let Some(pkt) = self
            .reader
            .read_packet()
            .context("ogg packet read after indexed seek")?
        {
            if !reached_key_page {
                // PacketReader drains all completed packets from a page before
                // reading the next page. Its reader position is the end of the
                // page completing this packet, including for spanning packets.
                // Back-pages can complete unrelated packets as well as start
                // the key packet; tona3 discards those before numbering packets
                // from the key page's top_packet_no.
                if self.reader.get_mut().stream_position()? <= key_page_file_offset {
                    continue;
                }
                reached_key_page = true;
            }
            if pkt.stream_serial() != self.video_serial {
                continue;
            }
            if theora_rs::th_packet_isheader(theora_rs::Packet::new(&pkt.data)) {
                continue;
            }

            if data_packet_no < key_frame_packet_no {
                data_packet_no = data_packet_no.saturating_add(1);
                continue;
            }
            if data_packet_no > target_packet_no {
                bail!(
                    "indexed Theora seek passed target packet {}",
                    target_packet_no
                );
            }
            if data_packet_no == key_frame_packet_no
                && theora_rs::th_packet_iskeyframe(theora_rs::Packet::new(&pkt.data)) != 1
            {
                bail!(
                    "indexed Theora packet {} is not a key frame",
                    key_frame_packet_no
                );
            }

            let b_o_s = pkt.first_in_stream();
            let e_o_s = pkt.last_in_stream();
            let granulepos = pkt.absgp_page() as i64;
            let data = pkt.data;
            let packetno = self
                .header_packet_count
                .saturating_add(data_packet_no as i64);
            let op = OggPacket {
                packet: data,
                b_o_s,
                e_o_s,
                granulepos,
                packetno,
            };
            self.decoder.frame_available = false;
            theora_rs::th_decode_packetin(&mut self.decoder, &op)?;
            self.packet_no = packetno.saturating_add(1);

            if data_packet_no == target_packet_no {
                if self.decoder.has_decoded_frame() {
                    let ycbcr = theora_rs::th_decode_ycbcr_out(&self.decoder)?;
                    return Ok(Some(pack_theorafile_frame(&self.raw_info, &ycbcr)?));
                }
                return Ok(None);
            }
            data_packet_no = data_packet_no.saturating_add(1);
        }

        bail!(
            "indexed Theora seek ended before target packet {}",
            target_packet_no
        )
    }

}

fn video_info_from_theora_info(info: &theora_rs::Info) -> VideoInfo {
    let width = info.pic_width.max(1) as i32;
    let height = info.pic_height.max(1) as i32;
    let fps = if info.fps_denominator != 0 {
        info.fps_numerator as f64 / info.fps_denominator as f64
    } else {
        0.0
    };
    let fmt = match info.pixel_fmt {
        PixelFmt::Pf420 | PixelFmt::Reserved => TH_PF_420,
        PixelFmt::Pf422 => TH_PF_422,
        PixelFmt::Pf444 => TH_PF_444,
    };
    VideoInfo {
        width,
        height,
        fps,
        fmt,
    }
}

pub fn decode_first_video_frame_from_memory(data: Vec<u8>) -> Result<(VideoInfo, Vec<u8>)> {
    let streams = read_ogg_streams(data).context("parse ogg packets")?;
    let video_serial = find_stream_by_magic(&streams, b"theora", 0x80)
        .ok_or_else(|| anyhow!("no video stream in ogg"))?;
    decode_theora_first_frame(
        streams
            .get(&video_serial)
            .ok_or_else(|| anyhow!("selected Theora stream missing"))?,
    )
    .context("decode first theora frame")
}

impl TheoraFile {
    pub fn open_from_memory(data: Vec<u8>) -> Result<Self> {
        let streams = read_ogg_streams(data).context("parse ogg packets")?;
        let video_serial = find_stream_by_magic(&streams, b"theora", 0x80)
            .ok_or_else(|| anyhow!("no video stream in ogg"))?;
        let audio_serial = find_stream_by_magic(&streams, b"vorbis", 0x01);

        let (info, video_frames) = decode_theora_stream(
            streams
                .get(&video_serial)
                .ok_or_else(|| anyhow!("selected Theora stream missing"))?,
        )
        .context("decode theora stream")?;

        let (has_audio_stream, audio_channels, audio_sample_rate, audio_samples) =
            if let Some(serial) = audio_serial {
                let (channels, sample_rate, samples) = decode_vorbis_stream(
                    streams
                        .get(&serial)
                        .ok_or_else(|| anyhow!("selected Vorbis stream missing"))?,
                )
                .context("decode vorbis stream")?;
                (true, channels, sample_rate, samples)
            } else {
                (false, 0, 0, Vec::new())
            };

        Ok(Self {
            info,
            video_frames,
            video_cursor: 0,
            audio_channels,
            audio_sample_rate,
            has_audio_stream,
            audio_samples,
            audio_cursor: 0,
        })
    }

    pub fn info(&self) -> VideoInfo {
        self.info
    }

    pub fn has_audio(&self) -> bool {
        self.has_audio_stream
    }

    pub fn audio_info(&self) -> Option<(i32, i32)> {
        if !self.has_audio_stream || self.audio_channels <= 0 || self.audio_sample_rate <= 0 {
            return None;
        }
        Some((self.audio_channels, self.audio_sample_rate))
    }

    pub fn reset(&mut self) {
        self.video_cursor = 0;
        self.audio_cursor = 0;
    }

    pub fn read_video_frame(&mut self, out: &mut [u8]) -> Result<bool> {
        let Some(frame) = self.video_frames.get(self.video_cursor) else {
            return Ok(false);
        };
        if out.len() < frame.len() {
            bail!(
                "video output buffer too small: need {} bytes, got {} bytes",
                frame.len(),
                out.len()
            );
        }
        out[..frame.len()].copy_from_slice(frame);
        self.video_cursor += 1;
        Ok(true)
    }

    pub fn read_audio_samples(&mut self, out: &mut [f32]) -> Result<usize> {
        if out.is_empty() || self.audio_cursor >= self.audio_samples.len() {
            return Ok(0);
        }
        let remaining = self.audio_samples.len() - self.audio_cursor;
        let count = remaining.min(out.len());
        out[..count]
            .copy_from_slice(&self.audio_samples[self.audio_cursor..self.audio_cursor + count]);
        self.audio_cursor += count;
        Ok(count)
    }
}

fn read_ogg_streams(data: Vec<u8>) -> Result<BTreeMap<u32, LogicalStream>> {
    let mut reader = PacketReader::new(Cursor::new(data));
    let mut streams = BTreeMap::<u32, LogicalStream>::new();
    let mut packetno_by_serial = BTreeMap::<u32, i64>::new();

    while let Some(pkt) = reader.read_packet().context("ogg packet read")? {
        let serial = pkt.stream_serial();
        let b_o_s = pkt.first_in_stream();
        let e_o_s = pkt.last_in_stream();
        let granulepos = pkt.absgp_page() as i64;
        let data = pkt.data;
        let packetno = packetno_by_serial.entry(serial).or_insert(0);
        let stream = streams.entry(serial).or_default();
        stream.packets.push(OggPacketMeta {
            data,
            b_o_s,
            e_o_s,
            granulepos,
            packetno: *packetno,
        });
        *packetno += 1;
    }

    Ok(streams)
}

fn find_stream_by_magic(
    streams: &BTreeMap<u32, LogicalStream>,
    magic: &[u8; 6],
    marker: u8,
) -> Option<u32> {
    streams.iter().find_map(|(&serial, stream)| {
        let first = stream.packets.first()?;
        if first.b_o_s
            && first.data.len() >= 7
            && first.data[0] == marker
            && &first.data[1..7] == magic
        {
            Some(serial)
        } else {
            None
        }
    })
}

fn decode_theora_first_frame(stream: &LogicalStream) -> Result<(VideoInfo, Vec<u8>)> {
    let mut parser = HeaderParser::new();
    let mut decoder = None;
    let mut first_frame = None;

    for pkt in &stream.packets {
        let op = OggPacket {
            packet: pkt.data.clone(),
            b_o_s: pkt.b_o_s,
            e_o_s: pkt.e_o_s,
            granulepos: pkt.granulepos,
            packetno: pkt.packetno,
        };

        if decoder.is_none() {
            let _ = parser.push(&op)?;
            if parser.is_ready() {
                decoder = Some(parser.decoder()?);
            }
            continue;
        }

        let dec = decoder.as_mut().expect("decoder allocated after headers");
        theora_rs::th_decode_packetin(dec, &op)?;
        if dec.has_decoded_frame() {
            let ycbcr = theora_rs::th_decode_ycbcr_out(dec)?;
            first_frame = Some(pack_theorafile_frame(&parser.info, &ycbcr)?);
            break;
        }
    }

    if !parser.is_ready() {
        bail!("stream ended before all Theora headers were parsed");
    }

    let info = &parser.info;
    let width = info.pic_width.max(1) as i32;
    let height = info.pic_height.max(1) as i32;
    let fps = if info.fps_denominator != 0 {
        info.fps_numerator as f64 / info.fps_denominator as f64
    } else {
        0.0
    };
    let fmt = match info.pixel_fmt {
        PixelFmt::Pf420 | PixelFmt::Reserved => TH_PF_420,
        PixelFmt::Pf422 => TH_PF_422,
        PixelFmt::Pf444 => TH_PF_444,
    };

    let frame = first_frame.ok_or_else(|| anyhow!("no decoded Theora frame in stream"))?;
    Ok((
        VideoInfo {
            width,
            height,
            fps,
            fmt,
        },
        frame,
    ))
}

fn decode_theora_stream(stream: &LogicalStream) -> Result<(VideoInfo, Vec<Vec<u8>>)> {
    let mut parser = HeaderParser::new();
    let mut decoder = None;
    let mut frames = Vec::<Vec<u8>>::new();

    for pkt in &stream.packets {
        let op = OggPacket {
            packet: pkt.data.clone(),
            b_o_s: pkt.b_o_s,
            e_o_s: pkt.e_o_s,
            granulepos: pkt.granulepos,
            packetno: pkt.packetno,
        };

        if decoder.is_none() {
            let _ = parser.push(&op)?;
            if parser.is_ready() {
                decoder = Some(parser.decoder()?);
            }
            continue;
        }

        let dec = decoder.as_mut().expect("decoder allocated after headers");
        theora_rs::th_decode_packetin(dec, &op)?;
        if dec.has_decoded_frame() {
            let ycbcr = theora_rs::th_decode_ycbcr_out(dec)?;
            frames.push(pack_theorafile_frame(&parser.info, &ycbcr)?);
        }
    }

    if !parser.is_ready() {
        bail!("stream ended before all Theora headers were parsed");
    }

    let info = &parser.info;
    let width = info.pic_width.max(1) as i32;
    let height = info.pic_height.max(1) as i32;
    let fps = if info.fps_denominator != 0 {
        info.fps_numerator as f64 / info.fps_denominator as f64
    } else {
        0.0
    };
    let fmt = match info.pixel_fmt {
        PixelFmt::Pf420 | PixelFmt::Reserved => TH_PF_420,
        PixelFmt::Pf422 => TH_PF_422,
        PixelFmt::Pf444 => TH_PF_444,
    };

    Ok((
        VideoInfo {
            width,
            height,
            fps,
            fmt,
        },
        frames,
    ))
}

fn pack_theorafile_frame(
    info: &theora_rs::Info,
    ycbcr: &theora_rs::YCbCrBuffer,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    let y_w = info.pic_width.max(1) as usize;
    let y_h = info.pic_height.max(1) as usize;
    copy_visible_plane(
        &mut out,
        &ycbcr[0],
        (info.pic_x & !1) as usize,
        (info.pic_y & !1) as usize,
        y_w,
        y_h,
    )?;

    let mut uv_w = y_w;
    let mut uv_h = y_h;
    let uv_x;
    let uv_y;
    match info.pixel_fmt {
        PixelFmt::Pf420 | PixelFmt::Reserved => {
            uv_w /= 2;
            uv_h /= 2;
            uv_x = (info.pic_x / 2) as usize;
            uv_y = (info.pic_y / 2) as usize;
        }
        PixelFmt::Pf422 => {
            uv_w /= 2;
            uv_x = (info.pic_x / 2) as usize;
            uv_y = (info.pic_y & !1) as usize;
        }
        PixelFmt::Pf444 => {
            uv_x = info.pic_x as usize;
            uv_y = info.pic_y as usize;
        }
    }

    copy_visible_plane(&mut out, &ycbcr[1], uv_x, uv_y, uv_w, uv_h)?;
    copy_visible_plane(&mut out, &ycbcr[2], uv_x, uv_y, uv_w, uv_h)?;
    Ok(out)
}

fn copy_visible_plane(
    dst: &mut Vec<u8>,
    plane: &theora_rs::ImgPlane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> Result<()> {
    let stride = plane.stride.max(0) as usize;
    let base = plane.data_offset;
    let needed = width.saturating_mul(height);
    dst.reserve(needed);
    for row in 0..height {
        let start = base
            .saturating_add((y + row).saturating_mul(stride))
            .saturating_add(x);
        let end = start.saturating_add(width);
        if end > plane.data.len() {
            bail!(
                "plane copy out of range: start={} end={} len={} stride={} width={} height={} offset={}",
                start,
                end,
                plane.data.len(),
                plane.stride,
                width,
                height,
                plane.data_offset
            );
        }
        dst.extend_from_slice(&plane.data[start..end]);
    }
    Ok(())
}

fn decode_vorbis_stream(stream: &LogicalStream) -> Result<(i32, i32, Vec<f32>)> {
    if stream.packets.len() < 3 {
        bail!("vorbis stream does not contain enough header packets");
    }

    let ident = read_header_ident(&stream.packets[0].data).context("read vorbis ident header")?;
    let _comment: CommentHeader =
        read_header_comment(&stream.packets[1].data).context("read vorbis comment header")?;
    let setup: SetupHeader = read_header_setup(
        &stream.packets[2].data,
        ident.audio_channels,
        (ident.blocksize_0, ident.blocksize_1),
    )
    .context("read vorbis setup header")?;

    let mut pwr = PreviousWindowRight::new();
    let mut samples = Vec::<f32>::new();
    for pkt in &stream.packets[3..] {
        let decoded: InterleavedSamples<f32> =
            read_audio_packet_generic(&ident, &setup, &pkt.data, &mut pwr)
                .context("decode vorbis audio packet")?;
        samples.extend_from_slice(&decoded.samples);
    }

    Ok((
        ident.audio_channels as i32,
        ident.audio_sample_rate as i32,
        samples,
    ))
}
