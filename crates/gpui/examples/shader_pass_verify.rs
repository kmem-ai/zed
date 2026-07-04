//! Offscreen eye-verify for the `ShaderPass` scene primitive (kcode #53 phase 2).
//!
//! Renders the naga-emitted GRADIENT shader (verbatim from the Wingman `shader` lib's `dump_msl`
//! dev tool) over a full window via [`Window::paint_shader_pass`], captures the frame headlessly,
//! and writes a PNG. Because the shader and the primitive share one binding contract, a correct
//! render proves the whole path: pipeline compile, the static fullscreen-quad vertex, the uniform
//! block at fragment `buffer(0)`, and the fragment entry `main_`.
//!
//! Expected image at `iTime = 0`. The Shadertoy body is
//! `fragColor = vec4(uv.x, uv.y, 0.5 + 0.5*sin(iTime), 1.0)` with `uv = fragCoord / iResolution.xy`,
//! and naga inserts the y-flip `fragCoord.y = iResolution.y - gl_FragCoord.y`, so:
//!   - red   ramps 0 (left)   -> 1 (right)
//!   - green ramps 0 (bottom) -> 1 (top)
//!   - blue  is a flat 0.5
//! => navy bottom-left, magenta bottom-right, green top-left, near-white top-right.
//!
//! Run (macOS only — the primitive is Metal-only):
//!   `cargo run -p gpui --example shader_pass_verify --features test-support`

use std::sync::Arc;

use gpui::{
    App, AppContext, Bounds, Context, HeadlessAppContext, IntoElement, NoopTextSystem,
    ParentElement, Pixels, Render, SharedString, Styled, Window, canvas, div, px, size,
};

/// The exact Metal source the Wingman `shader` lib emits for the canonical GRADIENT smoke shader
/// (regenerate with `cargo run -p shader --example dump_msl`). Kept verbatim so this example
/// eye-verifies naga's real output, not a hand-written stand-in.
const GRADIENT_MSL: &str = r#"// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _KcodeGlobals {
    metal::packed_float3 iResolution;
    float iTime;
    float iTimeDelta;
    int iFrame;
    char _pad4[8];
    metal::float4 iMouse;
    float iActivity;
    float iAttention;
    char _pad7[8];
    metal::packed_float3 iAccent;
    float iCompletion;
};
struct FragmentOutput {
    metal::float4 _kcode_fragColor;
};

void mainImage(
    thread metal::float4& fragColor,
    metal::float2 fragCoord,
    constant _KcodeGlobals& global
) {
    metal::float2 fragCoord_1 = {};
    metal::float2 uv = {};
    fragCoord_1 = fragCoord;
    metal::float2 _e30 = fragCoord_1;
    metal::float3 _e31 = global.iResolution;
    uv = _e30 / _e31.xy;
    metal::float2 _e35 = uv;
    metal::float2 _e37 = uv;
    float _e41 = global.iTime;
    fragColor = metal::float4(_e35.x, _e37.y, 0.5 + (0.5 * metal::sin(_e41)), 1.0);
    return;
}

void main_1(
    constant _KcodeGlobals& global,
    thread metal::float4& _kcode_fragColor,
    thread metal::float4& gl_FragCoord_1
) {
    metal::float4 _kcode_color = metal::float4(0.0);
    metal::float2 _kcode_fragCoord = {};
    metal::float4 _e31 = gl_FragCoord_1;
    metal::float3 _e33 = global.iResolution;
    metal::float4 _e35 = gl_FragCoord_1;
    _kcode_fragCoord = metal::float2(_e31.x, _e33.y - _e35.y);
    metal::float2 _e41 = _kcode_fragCoord;
    mainImage(_kcode_color, _e41, global);
    metal::float4 _e43 = _kcode_color;
    _kcode_fragColor = _e43;
    return;
}

struct main_Input {
};
struct main_Output {
    metal::float4 _kcode_fragColor [[color(0)]];
};
fragment main_Output main_(
  metal::float4 gl_FragCoord [[position]]
, constant _KcodeGlobals& global [[buffer(0)]]
) {
    metal::float4 _kcode_fragColor = {};
    metal::float4 gl_FragCoord_1 = {};
    gl_FragCoord_1 = gl_FragCoord;
    main_1(global, _kcode_fragColor, gl_FragCoord_1);
    metal::float4 _e39 = _kcode_fragColor;
    const auto _tmp = FragmentOutput {_e39};
    return main_Output { _tmp._kcode_fragColor };
}
"#;

/// Pack the 80-byte `ShaderUniforms` contract (mirrors the Wingman `shader` lib layout) with only
/// `iResolution` (bytes 0..12) and `iTime` (bytes 12..16) set; the rest are zero for this shader.
fn pack_uniforms(width: f32, height: f32, time: f32) -> Arc<[u8]> {
    let mut buf = [0u8; 80];
    buf[0..4].copy_from_slice(&width.to_ne_bytes());
    buf[4..8].copy_from_slice(&height.to_ne_bytes());
    buf[8..12].copy_from_slice(&1.0f32.to_ne_bytes()); // iResolution.z
    buf[12..16].copy_from_slice(&time.to_ne_bytes()); // iTime
    Arc::from(buf.to_vec().into_boxed_slice())
}

struct ShaderView;

impl Render for ShaderView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(
            canvas(
                |_bounds, _window, _cx| {},
                |bounds: Bounds<Pixels>, _prepaint, window: &mut Window, _cx: &mut App| {
                    let scale = window.scale_factor();
                    let w = f32::from(bounds.size.width) * scale;
                    let h = f32::from(bounds.size.height) * scale;
                    window.paint_shader_pass(
                        bounds,
                        1,
                        SharedString::from(GRADIENT_MSL),
                        SharedString::from("main_"),
                        pack_uniforms(w, h, 0.0),
                        false,
                    );
                },
            )
            .size_full(),
        )
    }
}

fn main() {
    let mut cx =
        HeadlessAppContext::with_platform(Arc::new(NoopTextSystem::new()), Arc::new(()), || {
            gpui_platform::current_headless_renderer()
        });
    let window = cx
        .open_window(size(px(512.0), px(512.0)), |_window, cx| {
            cx.new(|_| ShaderView)
        })
        .expect("open headless window");
    cx.run_until_parked();
    let img = cx
        .capture_screenshot(window.into())
        .expect("capture screenshot");
    let path = std::env::var_os("SHADER_VERIFY_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("shader_pass_verify.png"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    img.save(&path).expect("save png");
    println!(
        "wrote {} ({}x{})",
        path.display(),
        img.width(),
        img.height()
    );

    // Self-check the four corners so a broken render fails loudly instead of needing an eyeball.
    // Per the module docs the GRADIENT body maps red = uv.x (left->right), green = uv.y (bottom->top
    // after the naga y-flip), blue = 0.5 everywhere. A wrong uniform binding (garbage iResolution) or
    // a missing/mis-dispatched pass breaks these ranges. Sample a few px in from each corner.
    let (w, h) = (img.width(), img.height());
    let at = |x: u32, y: u32| {
        let p = img.get_pixel(x.min(w - 1), y.min(h - 1));
        (p[0], p[1], p[2])
    };
    let inset = 12;
    let checks = [
        (
            "bottom-left navy",
            at(inset, h - 1 - inset),
            (0u8, 60u8),
            (0u8, 60u8),
            (100u8, 160u8),
        ),
        (
            "bottom-right magenta",
            at(w - 1 - inset, h - 1 - inset),
            (200, 255),
            (0, 60),
            (100, 160),
        ),
        (
            "top-left green",
            at(inset, inset),
            (0, 60),
            (200, 255),
            (100, 160),
        ),
        (
            "top-right near-white",
            at(w - 1 - inset, inset),
            (200, 255),
            (200, 255),
            (100, 160),
        ),
    ];
    for (name, (r, g, b), (rlo, rhi), (glo, ghi), (blo, bhi)) in checks {
        assert!(
            (rlo..=rhi).contains(&r) && (glo..=ghi).contains(&g) && (blo..=bhi).contains(&b),
            "{name}: got rgb=({r},{g},{b}), want r in [{rlo},{rhi}] g in [{glo},{ghi}] b in [{blo},{bhi}]",
        );
    }
    println!("corner checks passed: gradient renders with the expected orientation and uniforms");
}
