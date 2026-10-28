// thermal_shutter_gate.h
//
// Header-only (OpenCV only). Detects the thermal camera's periodic shutter event (FFC) from the
// images themselves and tells the mapper which frames VIO must NOT use.
//
// What the shutter looks like (measured on a screen recording of /flir_boson/image_raw):
//   * the image FREEZES: the same frame is repeated for ~26 frames (~0.85 s at 30 fps), and
//   * the first frames after the freeze are slightly different in global brightness / noise
//     texture (offset correction), so a few "settle" frames are blocked as well.
//
// Detection = mean absolute difference to the previous frame:
//   diff < freeze_eps  -> frozen frame   (a real frame always differs by sensor noise, a repeated
//                                          frame differs by ~0)
// Blocked frames = frozen frames + settle_frames frames after the freeze ends.
//
// Time is not used for the decision (works on bag replay). Call update() for EVERY thermal frame,
// also while RGB is the active camera, so the "previous frame" is always the real previous one.
//
// Must be fed the same images the mapper would use (i.e. the topic that goes into FAST-LIVO2),
// not a different topic: a stabilizer between the camera and the mapper can change how a repeated
// frame looks (log `last().diff` first and check it drops to ~0 during the shutter).

#ifndef THERMAL_SHUTTER_GATE_H_
#define THERMAL_SHUTTER_GATE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct ShutterParams
{
  double freeze_eps = 0.02;  // mean abs gray-level difference below which a frame counts as repeated (real repeats: exactly 0)
  int settle_frames = 6;    // frames to keep blocked after the freeze ends (6 = 0.2 s at 30 fps)
  bool enabled = true;
};

struct ShutterSample
{
  double t = 0.0;
  double diff = 1e9;     // mean abs difference to the previous frame (gray levels)
  bool frozen = false;   // repeated frame
  bool blocked = false;  // VIO must not use this frame
  int run = 0;           // number of consecutive frozen frames so far
};

class ThermalShutterGate
{
public:
  explicit ThermalShutterGate(const ShutterParams &p = ShutterParams()) : p_(p) {}

  // Call once per thermal frame (t = image header time, seconds). Returns true when this frame
  // must NOT be used by the visual update.
  bool update(const cv::Mat &img, double t)
  {
    last_ = ShutterSample();
    last_.t = t;
    if (!p_.enabled || img.empty()) return false;

    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4) cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else gray = img;
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);

    if (prev_.empty() || prev_.size() != gray.size() || prev_.type() != gray.type())
    {
      prev_ = gray.clone();  // first frame (or camera changed): nothing to compare with
      run_ = 0;
      settle_left_ = 0;
      return false;
    }

    last_.diff = cv::norm(gray, prev_, cv::NORM_L1) / (static_cast<double>(gray.rows) * gray.cols);
    prev_ = gray.clone();
    last_.frozen = last_.diff < p_.freeze_eps;

    if (last_.frozen)
    {
      if (run_ == 0) ++event_count_;
      ++run_;
      settle_left_ = p_.settle_frames;
      last_.blocked = true;
    }
    else
    {
      run_ = 0;
      if (settle_left_ > 0)
      {
        --settle_left_;
        last_.blocked = true;
      }
    }
    last_.run = run_;
    return last_.blocked;
  }

  const ShutterSample &last() const { return last_; }
  int eventCount() const { return event_count_; }
  const ShutterParams &params() const { return p_; }

private:
  ShutterParams p_;
  ShutterSample last_;
  cv::Mat prev_;
  int run_ = 0;
  int settle_left_ = 0;
  int event_count_ = 0;
};

#endif  // THERMAL_SHUTTER_GATE_H_