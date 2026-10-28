// visual_mode_manager.h
//
// Header-only. Decides, from the RGB stream alone, whether FAST-LIVO2 should run on the
// RGB camera or on the thermal camera. It only depends on OpenCV.
//
// Time is always the IMAGE header time (seconds), never the wall clock, so a bag replay
// gives the same decisions every run.
//
// Metric (computed on a heavily downsampled gray image, which also averages out sensor noise):
//   dark pixel : gray < dark_level
//   flat pixel : gradient magnitude < flat_level   (no edges -> nothing for photometric alignment)
//   bad pixel  : dark OR flat
// A frame is "bad" when the fraction of bad pixels is high.
//
// State machine (RGB <-> THERMAL), with hysteresis so scenes near the threshold do not flip-flop:
//   RGB     -> THERMAL : (dark_frac > dark_enter OR flat_frac > flat_enter) for enter_time seconds
//   THERMAL -> RGB     : (dark_frac < dark_exit AND flat_frac < flat_exit)  for exit_time  seconds
//   and never more often than once per min_dwell seconds.
//
// select = "auto" | "rgb" | "thermal"   (the last two are the manual toggle)

#ifndef VISUAL_MODE_MANAGER_H_
#define VISUAL_MODE_MANAGER_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>

enum class VisualModality
{
  RGB = 0,
  THERMAL = 1
};

struct HealthParams
{
  int down_width = 80;        // analysis width in pixels; height follows the aspect ratio
  double dark_level = 35.0;   // gray level (0..255) below which a pixel is "dark"
  double flat_level = 4.0;    // gradient magnitude (gray levels per downsampled pixel) below which a pixel is "flat"
  // Separate thresholds for the two symptoms. Tuned on a real bag: dark_frac separates cleanly
  // (bright scenes: median 0.06, p99 0.44; pitch black: ~1.0). flat_frac does NOT yet (bright
  // scenes already reach 0.4-0.6 at flat_level 4), so the flat/smoke rule is disabled by
  // default: a value > 1.0 can never trigger. Enable it after tuning on smoke data.
  double dark_enter = 0.80;   // RGB -> THERMAL when dark_frac is above this ...
  double dark_exit = 0.50;    // THERMAL -> RGB only when dark_frac is below this ...
  double flat_enter = 2.0;    // ... or when flat_frac is above this (2.0 = disabled)
  double flat_exit = 2.0;     // ... and flat_frac is below this (2.0 = always satisfied)
  double enter_time = 1.0;    // enter condition must hold this many seconds
  double exit_time = 3.0;     // ... for this many seconds
  double min_dwell = 5.0;     // minimum seconds between two switches
  std::string select = "auto";
};

struct HealthSample
{
  double t = 0.0;
  double dark_frac = 0.0;
  double flat_frac = 0.0;
  double bad_frac = 0.0;
};

class VisualModeManager
{
public:
  explicit VisualModeManager(const HealthParams &p = HealthParams()) : p_(p) {}

  // Pure metric, no state. An empty image counts as completely bad (a dead RGB camera
  // should lead to thermal).
  static HealthSample evaluate(const cv::Mat &img, double t, const HealthParams &p)
  {
    HealthSample s;
    s.t = t;
    if (img.empty())
    {
      s.dark_frac = s.flat_frac = s.bad_frac = 1.0;
      return s;
    }

    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4) cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else gray = img;
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);

    const int w = std::max(8, p.down_width);
    const int h = std::max(8, static_cast<int>(std::lround(gray.rows * static_cast<double>(w) / gray.cols)));

    cv::Mat small8, small;
    cv::resize(gray, small8, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    small8.convertTo(small, CV_32F);

    // Sobel scaled by 1/8 gives the gradient in gray levels per pixel
    cv::Mat gx, gy, mag;
    cv::Sobel(small, gx, CV_32F, 1, 0, 3, 1.0 / 8.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(small, gy, CV_32F, 0, 1, 3, 1.0 / 8.0, 0.0, cv::BORDER_REPLICATE);
    cv::magnitude(gx, gy, mag);

    cv::Mat dark = small < p.dark_level;
    cv::Mat flat = mag < p.flat_level;
    cv::Mat bad = dark | flat;

    const double n = static_cast<double>(w) * h;
    s.dark_frac = cv::countNonZero(dark) / n;
    s.flat_frac = cv::countNonZero(flat) / n;
    s.bad_frac = cv::countNonZero(bad) / n;
    return s;
  }

  // Call once per RGB frame. Returns the modality that SHOULD be active now.
  // The mapper decides when it is safe to apply a change (between LIO/VIO pairs).
  VisualModality update(const cv::Mat &rgb_img, double t)
  {
    last_ = evaluate(rgb_img, t, p_);

    if (p_.select == "rgb")
    {
      mode_ = VisualModality::RGB;
      return mode_;
    }
    if (p_.select == "thermal")
    {
      mode_ = VisualModality::THERMAL;
      return mode_;
    }

    const bool dwell_ok = (t - last_switch_t_) >= p_.min_dwell;

    if (mode_ == VisualModality::RGB)
    {
      if (last_.dark_frac > p_.dark_enter || last_.flat_frac > p_.flat_enter)
      {
        if (!bad_active_)
        {
          bad_active_ = true;
          bad_since_ = t;
        }
        if (dwell_ok && (t - bad_since_) >= p_.enter_time) switchTo(VisualModality::THERMAL, t);
      }
      else
      {
        bad_active_ = false;
      }
    }
    else
    {
      if (last_.dark_frac < p_.dark_exit && last_.flat_frac < p_.flat_exit)
      {
        if (!good_active_)
        {
          good_active_ = true;
          good_since_ = t;
        }
        if (dwell_ok && (t - good_since_) >= p_.exit_time) switchTo(VisualModality::RGB, t);
      }
      else
      {
        good_active_ = false;
      }
    }
    return mode_;
  }

  VisualModality desired() const { return mode_; }
  const HealthSample &last() const { return last_; }
  int switchCount() const { return switch_count_; }
  const HealthParams &params() const { return p_; }

private:
  void switchTo(VisualModality m, double t)
  {
    mode_ = m;
    last_switch_t_ = t;
    bad_active_ = good_active_ = false;
    ++switch_count_;
  }

  HealthParams p_;
  HealthSample last_;
  VisualModality mode_ = VisualModality::RGB;
  bool bad_active_ = false, good_active_ = false;
  double bad_since_ = 0.0, good_since_ = 0.0;
  double last_switch_t_ = -1e18;
  int switch_count_ = 0;
};

#endif  // VISUAL_MODE_MANAGER_H_