#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
mod desktop {
    use std::{
        collections::VecDeque,
        io::{BufReader, Cursor},
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };

    use anyhow::{bail, Context};
    use kira::{
        manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings},
        sound::static_sound::{StaticSoundData, StaticSoundHandle},
    };
    use winit::{
        dpi::PhysicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    };

    use wmv_decoder::{
        asf::{AsfFile, AudioStreamInfo, VideoStreamInfo},
        AsfWmaDecoder, AsfWmv2Decoder, DecodedFrame,
    };

    pub fn run() -> anyhow::Result<()> {
        env_logger::init();
        let (input_path, video_only) = parse_input_arg()?;

        let (video_info, audio_info, duration_ms) = probe_streams(&input_path)?;
        let fourcc = String::from_utf8_lossy(&video_info.codec_four_cc);
        eprintln!(
            "[probe] video={} {}x{} stream={} duration={} ms",
            fourcc,
            video_info.width,
            video_info.height,
            video_info.stream_number,
            duration_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".into())
        );
        if let Some(audio) = &audio_info {
            eprintln!(
                "[probe] audio=0x{:04x} {} Hz {} ch stream={}",
                audio.format_tag, audio.sample_rate, audio.channels, audio.stream_number
            );
        } else {
            eprintln!("[probe] no supported WMA audio stream");
        }

        let video_w = video_info.width;
        let video_h = video_info.height;
        if video_w == 0 || video_h == 0 {
            bail!("ASF reports an invalid {}x{} video size", video_w, video_h);
        }

        // Create the window before touching the codec. A decoder hang/error must
        // never prevent the diagnostic UI from appearing.
        eprintln!("[ui] creating {}x{} window", video_w, video_h);
        let event_loop = EventLoop::new().context("create event loop")?;
        let window = WindowBuilder::new()
            .with_title(format!("wmv-player-wgpu - {}", input_path.display()))
            .with_inner_size(PhysicalSize::new(video_w, video_h))
            .build(&event_loop)
            .context("create player window")?;

        eprintln!("[ui] initializing wgpu");
        let renderer = pollster::block_on(Renderer::new(window, video_w, video_h))
            .context("initialize wgpu renderer")?;
        eprintln!("[ui] window ready; starting codec workers");

        let stop = Arc::new(AtomicBool::new(false));
        let (video_tx, video_rx) = crossbeam_channel::bounded::<VideoEvent>(8);
        let (audio_tx, audio_rx) = crossbeam_channel::bounded::<AudioEvent>(1);

        // Kira's StaticSoundData path needs the complete PCM buffer. Do not run
        // that full-file WMA decode concurrently with WMV3: on a 1080p movie it
        // needlessly steals an entire CPU core from the video decoder. The
        // window remains responsive while audio is prepared; once ready we start
        // video and start Kira on the first decoded video frame. --video-only is
        // available when the goal is pure WMV3 profiling.
        let audio_predecode = audio_info.is_some() && !video_only;
        let _audio_thread = if audio_predecode {
            Some(spawn_audio_decode_thread(input_path.clone(), audio_tx, stop.clone()))
        } else {
            None
        };

        let mut state = PlayerState {
            renderer,
            video_tx,
            video_rx,
            audio_rx,
            stop,
            input_path,
            video_started: false,
            pending_audio: None,
            _audio: None,
            pending_video: VecDeque::new(),
            video_clock: None,
            presented_frames: 0,
            dropped_frames: 0,
        };
        // Video must never wait for a full-file WMA predecode.  The previous
        // diagnostic-player revision only started WMV decoding after the audio
        // worker had decoded the entire movie, leaving the window blank for the
        // whole predecode interval.  Start video immediately; audio preparation
        // is independent and may finish later.
        state.start_video_decode_if_needed();

        event_loop.run(move |event, elwt| {
            // Do not spin the UI thread at 100% CPU. Crossbeam does not wake
            // winit directly, so use a short timed wait; 4 ms is comfortably
            // below a 60 Hz presentation interval without busy-polling.
            elwt.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(4)));
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        state.stop.store(true, Ordering::Relaxed);
                        elwt.exit();
                    }
                    WindowEvent::Resized(size) => {
                        state.renderer.resize(size.width, size.height);
                    }
                    WindowEvent::RedrawRequested => {
                        while let Ok(event) = state.audio_rx.try_recv() {
                            match event {
                                AudioEvent::Decoded(audio) => {
                                    eprintln!("[audio] predecode complete");
                                    state.pending_audio = Some(audio);
                                    state.start_audio_if_ready();
                                }
                                AudioEvent::NoAudio => {
                                    eprintln!("[audio] decoder produced no PCM frames");
                                }
                                AudioEvent::Error(err) => {
                                    eprintln!("[audio] decode failed: {err:#}");
                                }
                            }
                        }

                        // Present anything that is already due before receiving more
                        // decoded frames.  Otherwise a fast decoder can keep try_recv()
                        // continuously successful and starve presentation entirely.
                        let mut needs_render = state.present_due_video_frame();
                        state.collect_video_events();
                        // A first frame (or another due frame) may have arrived in the
                        // bounded receive step above, so give the presenter one more
                        // chance in the same redraw.
                        needs_render |= state.present_due_video_frame();
                        if needs_render {
                            if let Err(err) = state.renderer.render() {
                                eprintln!("[wgpu] render failed: {err:#}");
                                state.stop.store(true, Ordering::Relaxed);
                                elwt.exit();
                            }
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    state.renderer.request_redraw();
                }
                Event::LoopExiting => {
                    state.stop.store(true, Ordering::Relaxed);
                }
                _ => {}
            }
        })?;

        Ok(())
    }

    fn parse_input_arg() -> anyhow::Result<(PathBuf, bool)> {
        let mut args = std::env::args_os().skip(1);
        let mut input = None;
        let mut video_only = false;
        while let Some(arg) = args.next() {
            if arg == "--input" {
                let value = args.next().context("--input requires a file path")?;
                input = Some(PathBuf::from(value));
            } else if arg == "--video-only" {
                video_only = true;
            } else if arg == "-h" || arg == "--help" {
                println!("Usage: wmv-player-wgpu --input <file.wmv> [--video-only]");
                std::process::exit(0);
            } else {
                bail!(
                    "unknown argument {:?}\nUsage: wmv-player-wgpu --input <file.wmv> [--video-only]",
                    arg
                );
            }
        }
        let path = input.context("missing --input <file.wmv>")?;
        if !path.is_file() {
            bail!("input file does not exist: {}", path.display());
        }
        Ok((path, video_only))
    }

    fn probe_streams(
        path: &Path,
    ) -> anyhow::Result<(VideoStreamInfo, Option<AudioStreamInfo>, Option<u64>)> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let asf = AsfFile::open(&mut reader).context("parse ASF header")?;
        let video = asf
            .video_streams
            .iter()
            .find(|v| {
                matches!(
                    String::from_utf8_lossy(&v.codec_four_cc).to_ascii_uppercase().as_str(),
                    "WMV1" | "WMV2" | "WMV3"
                )
            })
            .cloned()
            .context("no supported WMV1/WMV2/WMV3 video stream")?;
        let audio = asf
            .audio_streams
            .iter()
            .find(|a| matches!(a.format_tag, 0x0160 | 0x0161 | 0x0162))
            .cloned();
        Ok((video, audio, asf.play_duration_ms))
    }

    struct PlayerState {
        renderer: Renderer,
        video_tx: crossbeam_channel::Sender<VideoEvent>,
        video_rx: crossbeam_channel::Receiver<VideoEvent>,
        audio_rx: crossbeam_channel::Receiver<AudioEvent>,
        stop: Arc<AtomicBool>,
        input_path: PathBuf,
        video_started: bool,
        pending_audio: Option<DecodedAudio>,
        _audio: Option<AudioPlayback>,
        pending_video: VecDeque<DecodedFrame>,
        video_clock: Option<VideoClock>,
        presented_frames: usize,
        dropped_frames: usize,
    }

    #[derive(Clone, Copy)]
    struct VideoClock {
        first_pts_ms: u32,
        start: Instant,
    }

    impl VideoClock {
        fn target_instant(self, pts_ms: u32) -> Instant {
            self.start
                + Duration::from_millis(pts_ms.saturating_sub(self.first_pts_ms) as u64)
        }

        fn media_time_ms(self, now: Instant) -> u64 {
            now.saturating_duration_since(self.start).as_millis() as u64
                + self.first_pts_ms as u64
        }
    }

    impl PlayerState {
        fn start_video_decode_if_needed(&mut self) {
            if self.video_started {
                return;
            }
            self.video_started = true;
            let _ = spawn_video_decode_thread(
                self.input_path.clone(),
                self.video_tx.clone(),
                self.stop.clone(),
            );
        }

        fn start_audio_if_ready(&mut self) {
            if self._audio.is_some() || self.video_clock.is_none() {
                return;
            }
            let Some(audio) = self.pending_audio.take() else {
                return;
            };
            match AudioPlayback::start(audio) {
                Ok(playback) => {
                    eprintln!("[audio] Kira playback started");
                    self._audio = Some(playback);
                }
                Err(err) => {
                    eprintln!("[audio] Kira output init/play failed: {err:#}");
                }
            }
        }

        fn collect_video_events(&mut self) {
            // Keep decode-ahead genuinely bounded.  The channel itself is bounded,
            // but draining it with an unbounded `while try_recv()` moved every frame
            // into `pending_video`, defeating backpressure.  Once the release build
            // made WMV3 faster this could keep the UI inside this function long
            // enough that the already-uploaded picture appeared frozen.
            const MAX_PENDING_VIDEO: usize = 8;
            let mut budget = MAX_PENDING_VIDEO.saturating_sub(self.pending_video.len());
            while budget != 0 {
                let Ok(event) = self.video_rx.try_recv() else {
                    break;
                };
                match event {
                    VideoEvent::Frame(frame) => {
                        // WMV3 with B pictures is decoded in coded order, while
                        // PTS is presentation order. Keep the small decode-ahead
                        // queue sorted by PTS instead of assuming monotonic PTS.
                        let pos = self
                            .pending_video
                            .iter()
                            .position(|queued| queued.pts_ms > frame.pts_ms)
                            .unwrap_or(self.pending_video.len());
                        self.pending_video.insert(pos, frame);
                        budget -= 1;
                    }
                    VideoEvent::Eof { frames, last_pts_ms } => {
                        eprintln!(
                            "[video] EOF: decoded {} frames, last pts={} ms",
                            frames, last_pts_ms
                        );
                    }
                    VideoEvent::Error { frames, error } => {
                        eprintln!(
                            "[video] decode failed after {} frames: {:#}",
                            frames, error
                        );
                    }
                }
            }
        }

        fn present_due_video_frame(&mut self) -> bool {
            if self.video_clock.is_none() {
                let Some(first) = self.pending_video.front() else {
                    return false;
                };
                self.video_clock = Some(VideoClock {
                    first_pts_ms: first.pts_ms,
                    start: Instant::now(),
                });
                eprintln!(
                    "[video] presentation clock started at pts={} ms",
                    first.pts_ms
                );
                self.start_audio_if_ready();
            }

            let clock = self.video_clock.expect("video clock initialized");
            let now = Instant::now();
            let mut due = None::<DecodedFrame>;
            let mut due_count = 0usize;

            while let Some(front) = self.pending_video.front() {
                if clock.target_instant(front.pts_ms) > now {
                    break;
                }
                due = self.pending_video.pop_front();
                due_count += 1;
            }

            let Some(frame) = due else {
                return false;
            };

            // If several frames became due between redraws, only the newest one
            // can affect the display. Dropping the older late frames is what a
            // real-time video presenter does; uploading them all in one redraw
            // would merely flash through stale frames as fast as the CPU allows.
            if due_count > 1 {
                self.dropped_frames += due_count - 1;
                eprintln!(
                    "[video] presentation late: dropped {} stale frame(s), total_dropped={}",
                    due_count - 1,
                    self.dropped_frames
                );
            }

            let index = self.presented_frames;
            if index < 5 || index % 60 == 0 {
                print_video_frame_diag(index, &frame);
            }
            let media_now = clock.media_time_ms(now);
            let lateness_ms = media_now.saturating_sub(frame.pts_ms as u64);
            eprintln!(
                "[video] present frame={} pts={} ms clock={} ms late={} ms queued={}",
                index,
                frame.pts_ms,
                media_now,
                lateness_ms,
                self.pending_video.len()
            );
            self.presented_frames += 1;
            self.renderer.upload_decoded_frame(&frame);
            true
        }
    }

    enum VideoEvent {
        Frame(DecodedFrame),
        Eof { frames: usize, last_pts_ms: u32 },
        Error { frames: usize, error: anyhow::Error },
    }

    fn spawn_video_decode_thread(
        path: PathBuf,
        tx: crossbeam_channel::Sender<VideoEvent>,
        stop: Arc<AtomicBool>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            eprintln!("[video] opening decoder: {}", path.display());
            let file = match std::fs::File::open(&path) {
                Ok(file) => file,
                Err(error) => {
                    let _ = tx.send(VideoEvent::Error {
                        frames: 0,
                        error: anyhow::Error::new(error).context("open WMV input"),
                    });
                    return;
                }
            };
            let mut decoder = match AsfWmv2Decoder::open(BufReader::new(file)) {
                Ok(decoder) => decoder,
                Err(error) => {
                    let _ = tx.send(VideoEvent::Error {
                        frames: 0,
                        error: anyhow::Error::new(error).context("open ASF/WMV decoder"),
                    });
                    return;
                }
            };
            eprintln!("[video] decoder opened; requesting first frame");

            let mut frames = 0usize;
            let mut last_pts_ms = 0u32;

            while !stop.load(Ordering::Relaxed) {
                let decode_started = Instant::now();
                if frames < 5 || frames % 60 == 0 {
                    eprintln!("[video] next_frame begin index={frames}");
                }
                let frame = match decoder.next_frame() {
                    Ok(Some(frame)) => frame,
                    Ok(None) => {
                        let _ = tx.send(VideoEvent::Eof { frames, last_pts_ms });
                        break;
                    }
                    Err(error) => {
                        let _ = tx.send(VideoEvent::Error {
                            frames,
                            error: anyhow::Error::new(error),
                        });
                        break;
                    }
                };
                let decode_elapsed = decode_started.elapsed();
                if frames < 5 || frames % 60 == 0 || decode_elapsed >= Duration::from_millis(50) {
                    eprintln!(
                        "[video] next_frame end index={} pts={} key={} decode_ms={:.2}",
                        frames,
                        frame.pts_ms,
                        frame.is_key_frame,
                        decode_elapsed.as_secs_f64() * 1000.0
                    );
                }

                // Decoding and presentation are intentionally decoupled. The
                // bounded channel below is the decode-ahead queue; the UI thread
                // owns the PTS clock and decides exactly when a frame becomes
                // visible.
                last_pts_ms = frame.pts_ms;
                frames += 1;
                if tx.send(VideoEvent::Frame(frame)).is_err() {
                    break;
                }
            }
        })
    }

    fn print_video_frame_diag(index: usize, frame: &DecodedFrame) {
        let (y_min, y_max) = min_max(&frame.frame.y);
        let (u_min, u_max) = min_max(&frame.frame.cb);
        let (v_min, v_max) = min_max(&frame.frame.cr);
        eprintln!(
            "[video] frame={} pts={} ms key={} {}x{} Y={}..{} U={}..{} V={}..{}",
            index,
            frame.pts_ms,
            frame.is_key_frame,
            frame.frame.width,
            frame.frame.height,
            y_min,
            y_max,
            u_min,
            u_max,
            v_min,
            v_max
        );
    }

    fn min_max(data: &[u8]) -> (u8, u8) {
        let mut min = u8::MAX;
        let mut max = u8::MIN;
        for &v in data {
            min = min.min(v);
            max = max.max(v);
        }
        if data.is_empty() { (0, 0) } else { (min, max) }
    }

    enum AudioEvent {
        Decoded(DecodedAudio),
        NoAudio,
        Error(anyhow::Error),
    }

    fn spawn_audio_decode_thread(
        path: PathBuf,
        tx: crossbeam_channel::Sender<AudioEvent>,
        stop: Arc<AtomicBool>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            eprintln!("[audio] opening decoder: {}", path.display());
            match decode_audio_stereo(&path) {
                Ok(Some(audio)) => {
                    let _ = tx.send(AudioEvent::Decoded(audio));
                }
                Ok(None) => {
                    let _ = tx.send(AudioEvent::NoAudio);
                }
                Err(err) => {
                    let _ = tx.send(AudioEvent::Error(err));
                }
            }
        })
    }

    struct DecodedAudio {
        sample_rate: u32,
        stereo_samples: Vec<f32>,
    }

    fn decode_audio_stereo(path: &Path) -> anyhow::Result<Option<DecodedAudio>> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open {} for audio", path.display()))?;
        let mut decoder = AsfWmaDecoder::open(BufReader::new(file))
            .context("open ASF/WMA decoder")?;
        let sample_rate = decoder.sample_rate();
        let channels = decoder.channels() as usize;
        if sample_rate == 0 || channels == 0 {
            bail!("invalid audio format: {} Hz, {} channels", sample_rate, channels);
        }

        let mut stereo = Vec::<f32>::new();
        let mut first_audio_pts = None;
        let mut frames = 0usize;
        loop {
            let decoded = match decoder.next_frame() {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(err) => return Err(anyhow::Error::new(err).context("decode WMA packet")),
            };
            first_audio_pts.get_or_insert(decoded.pts_ms);
            frames += 1;
            append_stereo(&mut stereo, &decoded.frame.samples, channels);
        }

        if stereo.is_empty() {
            eprintln!("[audio] decoder produced no PCM frames");
            return Ok(None);
        }

        let first_audio_pts = first_audio_pts.unwrap_or(0);
        eprintln!(
            "[audio] decoded {} chunks, {} Hz, {} ch -> stereo, first pts={} ms, samples={} ({:.3} s)",
            frames,
            sample_rate,
            channels,
            first_audio_pts,
            stereo.len(),
            stereo.len() as f64 / 2.0 / sample_rate as f64
        );
        Ok(Some(DecodedAudio {
            sample_rate,
            stereo_samples: stereo,
        }))
    }

    fn append_stereo(dst: &mut Vec<f32>, samples: &[f32], channels: usize) {
        if channels == 1 {
            dst.reserve(samples.len().saturating_mul(2));
            for &s in samples {
                dst.push(s);
                dst.push(s);
            }
            return;
        }
        let frames = samples.len() / channels;
        dst.reserve(frames.saturating_mul(2));
        for frame in 0..frames {
            let base = frame * channels;
            dst.push(samples[base]);
            dst.push(samples[base + 1]);
        }
    }


    struct AudioPlayback {
        _manager: AudioManager<DefaultBackend>,
        _handle: StaticSoundHandle,
    }

    impl AudioPlayback {
        fn start(audio: DecodedAudio) -> anyhow::Result<Self> {
            let wav = encode_wav_stereo_i16(&audio.stereo_samples, audio.sample_rate);
            let sound = StaticSoundData::from_cursor(Cursor::new(wav))
                .context("Kira decode generated PCM WAV")?;
            let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
                .context("create Kira audio manager")?;
            let handle = manager.play(sound).context("start Kira movie audio")?;
            Ok(Self {
                _manager: manager,
                _handle: handle,
            })
        }
    }

    fn encode_wav_stereo_i16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let data_len = samples.len().saturating_mul(2) as u32;
        let mut out = Vec::with_capacity(44usize.saturating_add(data_len as usize));
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&sample_rate.saturating_mul(4).to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for &sample in samples {
            let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    // RENDERER_START
    struct Renderer {
        // IMPORTANT: surface must drop before window.
        surface: wgpu::Surface<'static>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: wgpu::SurfaceConfiguration,

        pipeline: wgpu::RenderPipeline,
        sampler: wgpu::Sampler,

        tex_y: wgpu::Texture,
        tex_u: wgpu::Texture,
        tex_v: wgpu::Texture,
        bind_group: wgpu::BindGroup,

        // Staging buffers for row-padding (wgpu requires bytes_per_row alignment).
        scratch_y: Vec<u8>,
        scratch_u: Vec<u8>,
        scratch_v: Vec<u8>,

        video_w: u32,
        video_h: u32,
        has_frame: bool,
        window: Window,
    }

    impl Renderer {
        async fn new(window: Window, video_w: u32, video_h: u32) -> anyhow::Result<Self> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let surface_tmp = instance.create_surface(&window)?;
            let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface_tmp) };

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| anyhow::anyhow!("no suitable GPU adapter"))?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await?;

            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats[0];

            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: caps.present_modes[0],
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("yuv_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            let (tex_y, view_y) = Self::make_plane_tex(&device, video_w, video_h, "Y");
            let (tex_u, view_u) = Self::make_plane_tex(&device, video_w / 2, video_h / 2, "U");
            let (tex_v, view_v) = Self::make_plane_tex(&device, video_w / 2, video_h / 2, "V");

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("yuv_shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("yuv_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yuv_bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_y),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_u),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view_v),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("yuv_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("yuv_pipe"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            Ok(Self {
                surface,
                device,
                queue,
                config,
                pipeline,
                sampler,
                tex_y,
                tex_u,
                tex_v,
                bind_group,

                scratch_y: Vec::new(),
                scratch_u: Vec::new(),
                scratch_v: Vec::new(),
                video_w,
                video_h,
                has_frame: false,
                window,
            })
        }

        fn request_redraw(&self) {
            self.window.request_redraw();
        }

        fn resize(&mut self, w: u32, h: u32) {
            let w = w.max(1);
            let h = h.max(1);
            if self.config.width == w && self.config.height == h {
                return;
            }
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }

        fn upload_decoded_frame(&mut self, decoded: &DecodedFrame) {
            let vf = &decoded.frame;
            if vf.width != self.video_w || vf.height != self.video_h {
                return;
            }
            let w = self.video_w;
            let h = self.video_h;
            self.has_frame = true;

            Self::upload_plane_static(&self.queue, &self.tex_y, w, h, &vf.y, &mut self.scratch_y);
            Self::upload_plane_static(
                &self.queue,
                &self.tex_u,
                w / 2,
                h / 2,
                &vf.cb,
                &mut self.scratch_u,
            );
            Self::upload_plane_static(
                &self.queue,
                &self.tex_v,
                w / 2,
                h / 2,
                &vf.cr,
                &mut self.scratch_v,
            );
        }

        fn upload_plane_static(
            queue: &wgpu::Queue,
            tex: &wgpu::Texture,
            w: u32,
            h: u32,
            data: &[u8],
            scratch: &mut Vec<u8>,
        ) {
            if w == 0 || h == 0 {
                return;
            }

            const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let stride = ((w + (ALIGN - 1)) / ALIGN) * ALIGN;

            let (src, bpr) = if stride == w {
                (data, w)
            } else {
                let needed = (stride as usize) * (h as usize);
                if scratch.len() < needed {
                    scratch.resize(needed, 0);
                }
                for row in 0..(h as usize) {
                    let dst0 = row * (stride as usize);
                    let src0 = row * (w as usize);
                    scratch[dst0..dst0 + (w as usize)]
                        .copy_from_slice(&data[src0..src0 + (w as usize)]);
                }
                (&scratch[..needed], stride)
            };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                src,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }

        fn render(&mut self) -> anyhow::Result<()> {
            let frame = match self.surface.get_current_texture() {
                Ok(f) => f,
                Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                    self.surface.configure(&self.device, &self.config);
                    return Ok(());
                }
                Err(wgpu::SurfaceError::Timeout) => return Ok(()),
                Err(e) => return Err(anyhow::anyhow!("surface error: {e:?}")),
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render_encoder"),
                });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if self.has_frame {
                    rpass.set_pipeline(&self.pipeline);
                    rpass.set_bind_group(0, &self.bind_group, &[]);
                    rpass.draw(0..4, 0..1);
                }
            }

            self.queue.submit(Some(encoder.finish()));
            frame.present();
            Ok(())
        }

        fn make_plane_tex(
            device: &wgpu::Device,
            w: u32,
            h: u32,
            label: &str,
        ) -> (wgpu::Texture, wgpu::TextureView) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("tex_{label}")),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        }
    }

    const SHADER: &str = r#"
    struct VSOut {
        @builtin(position) pos: vec4<f32>,
        @location(0) uv: vec2<f32>,
    };

    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
        var positions = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 1.0, -1.0),
            vec2<f32>(-1.0,  1.0),
            vec2<f32>( 1.0,  1.0),
        );
        var uvs = array<vec2<f32>, 4>(
            vec2<f32>(0.0, 1.0),
            vec2<f32>(1.0, 1.0),
            vec2<f32>(0.0, 0.0),
            vec2<f32>(1.0, 0.0),
        );

        var out: VSOut;
        out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
        out.uv = uvs[idx];
        return out;
    }

    @group(0) @binding(0) var tex_y: texture_2d<f32>;
    @group(0) @binding(1) var tex_u: texture_2d<f32>;
    @group(0) @binding(2) var tex_v: texture_2d<f32>;
    @group(0) @binding(3) var samp: sampler;

    fn yuv_to_rgb(y: f32, u: f32, v: f32) -> vec3<f32> {
        // Match siglus_scene_vm::movie::wmv_yuv_frame_to_rgba:
        // BT.601 limited-range YUV420 -> RGB.
        let c = max(y * 255.0 - 16.0, 0.0);
        let d = u * 255.0 - 128.0;
        let e = v * 255.0 - 128.0;
        let r = (298.0 * c + 409.0 * e + 128.0) / 256.0 / 255.0;
        let g = (298.0 * c - 100.0 * d - 208.0 * e + 128.0) / 256.0 / 255.0;
        let b = (298.0 * c + 516.0 * d + 128.0) / 256.0 / 255.0;
        return vec3<f32>(r, g, b);
    }

    @fragment
    fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
        let y = textureSample(tex_y, samp, in.uv).r;
        let u = textureSample(tex_u, samp, in.uv).r;
        let v = textureSample(tex_v, samp, in.uv).r;
        let rgb = clamp(yuv_to_rgb(y, u, v), vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb, 1.0);
    }
    "#;
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn main() {
    if let Err(err) = desktop::run() {
        eprintln!("wmv-player-wgpu: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn main() {
    // The real player implementation and all of its desktop-only dependencies
    // are cfg-gated above. Keep a tiny fallback main so `--all-targets` remains
    // well-formed on non-desktop targets.
    eprintln!("wmv-player-wgpu is only available on Linux, macOS, and Windows");
}
