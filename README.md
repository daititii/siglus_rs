# siglus_rs

<img src="./icon/Icon.png" alt="icon" align="left" width="100" style="margin-right: 10px;" />

**siglus_rs** is an unofficial Rust implementation and multi-platform port of SiglusEngine.

This project is non-commercial and intended for research purposes.

<br clear="left"/>


## Example screenshots
* siglus_rs on macOS
![On Mac](./images/screenshot.png)

* siglus_rs on iOS
![On iOS](./images/ios-screenshot.png)

* siglus_rs on WebAssembly
![On Web](./images/screenshot-wasm.png)

* siglus_rs works on a wide range of platforms, including Windows, Linux, macOS, iOS, Android, and WebAssembly.

| Platform | Targets |
|---|---|
| Linux | x86_64, aarch64 |
| FreeBSD | x86_64 |
| Windows | x86_64, ARM64 |
| macOS | DMG app bundle |
| iOS | arm64 device, arm64 simulator, x86_64 simulator |
| Android | arm64-v8a, x86_64 |
| WebAssembly | wasm32-unknown-unknown |

## Pre-built binaries
* See preview releases on [GitHub Releases](https://github.com/xmoezzz/siglus_rs/releases)

## Documentation Availability
* API documentation is available at [docs](https://xmoezzz.github.io/siglus_rs/)

## Run

```bash
cargo run --release -p siglus_scene_vm --bin siglus_engine -- --project-dir ~/Documents/siglus_rs-main/testcase
```

## Resource decryption key

SiglusEngine games require a secondary key to decrypt protected resources.

Create `key.toml` in the game root:

```toml
key = [0x00, 0x11, ...] # 16 bytes
```

For most trial versions, a secondary key is usually not required, and a 16-byte zero key is enough:

```toml
key = [
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
]
```

For full retail versions, a game-specific secondary key is usually required.

There are several practical ways to obtain the key:

1. Static extraction, when the game executable is not encrypted or obfuscated (recommended). The general idea can be found in this repository:

   https://github.com/xmoezzz/siglus_static_key_tool

2. Dynamic extraction. The general idea can be found in this older repository:

   https://github.com/xmoezzz/SiglusExtract

3. Known-key databases maintained by some extractor tools.

## License
This project is licensed under the MPL-2.0 License. See [LICENSE](./LICENSE-MPL-2.0) for details.
