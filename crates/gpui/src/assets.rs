use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use parking_lot::Mutex;
use std::{
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
        }
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        self.data.len()
    }
}

/// Images whose last `RenderImage` handle dropped and whose sprite-atlas tiles are still to be
/// released: `(id, frame_count)`. Drained by [`crate::App::release_dropped_images`], which every
/// [`crate::Window::draw`] runs before it paints.
static DROPPED_IMAGES: Mutex<Vec<(ImageId, usize)>> = Mutex::new(Vec::new());

/// Dropping the last handle to a `RenderImage` frees its CPU pixels here and queues its atlas tiles
/// for release on every window's next draw. Without this, the device copy an earlier
/// `Window::paint_image` uploaded outlived the image until a holder remembered to call
/// `App::drop_image` — and any holder that decoded more than once (a per-frame re-decode, a frame
/// stream, a cache eviction) leaked one texture per decode.
impl Drop for RenderImage {
    fn drop(&mut self) {
        if !self.data.is_empty() {
            DROPPED_IMAGES.lock().push((self.id, self.data.len()));
        }
    }
}

/// Take every image dropped since the previous call, for the atlas release.
pub(crate) fn take_dropped_images() -> Vec<(ImageId, usize)> {
    std::mem::take(&mut *DROPPED_IMAGES.lock())
}

/// How many dropped images still await the atlas release the next draw performs.
pub fn dropped_images_pending() -> usize {
    DROPPED_IMAGES.lock().len()
}

/// Whether the image with `id` has been dropped and still awaits its atlas release.
pub fn is_image_release_pending(id: ImageId) -> bool {
    DROPPED_IMAGES
        .lock()
        .iter()
        .any(|(dropped, _)| *dropped == id)
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn dropping_a_render_image_queues_its_atlas_release() {
        let frame = image::Frame::new(image::RgbaImage::new(1, 1));
        let image = RenderImage::new(vec![frame]);
        let id = image.id;
        assert!(!is_image_release_pending(id));
        drop(image);
        assert!(is_image_release_pending(id));
        assert!(take_dropped_images().contains(&(id, 1)));
        assert!(!is_image_release_pending(id));
    }

    #[test]
    fn dropping_a_frameless_render_image_queues_nothing() {
        let image = RenderImage::new(SmallVec::new());
        let id = image.id;
        drop(image);
        assert!(!is_image_release_pending(id));
    }

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }
}
