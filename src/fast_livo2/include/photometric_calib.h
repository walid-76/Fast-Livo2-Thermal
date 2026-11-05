// photometric_calib.h
//
// Online global gain/bias photometric calibration for an AGC thermal camera,
// following the simplified formulation of PL-LIT (Xia et al., arXiv 2606.29259,
// Sec. IV-C1, Eqs. 8-10), which itself streamlines Das et al. (RA-L 2021).
//
//   residual   r_k(g,b) = I_ref,k - ( g * I_cur,k + b )
//   cost       sum_k rho(r_k) + lambda1 * (g - 1)^2 + lambda2 * b^2
//   correction I_corr = g * I_raw + b          (g = effective gain "a-b" in the paper)
//
// The model is linear in (g,b), so the robust fit is a few IRLS iterations on a
// 2x2 system. No Ceres / Eigen / OpenCV needed: this header is dependency free.
//
// NOTE: the paper does not publish lambda1, lambda2 or the robust loss rho.
// The defaults below (Huber loss, lambda values) are my own starting points
// and MUST be tuned on your data.
//
// Usage per frame:
//   calib.clearSamples();
//   ... for each map point: calib.addSample(ref_patch_mean, raw_cur_patch_mean);
//   bool updated = calib.solve();       // updates calib.g, calib.b
//   calib.buildLUT(lut, calib.g, calib.b);   // apply to the 8-bit image

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct PhotometricCalibParams
{
  // Regularization, applied to the *mean* data term (the cost is divided by N), intensities in gray levels 0..255.
  // So these equal the paper's lambdas divided by N (the paper sums over correspondences without normalizing).
  // The gain g is only identifiable when the correspondences span a wide intensity range: the shrinkage toward g=1
  // is roughly lambda1 / var(I_cur). Thermal scenes often have a narrow spread, so keep lambda1 small and tune it.
  double lambda1 = 25.0;   // pulls effective gain g toward 1.
  double lambda2 = 0.01;   // pulls bias b toward 0 (gray levels^2 units).
  // Robust loss (Huber) threshold on the patch-mean residual, in gray levels. Too small and a large AGC jump gets
  // down-weighted as "outliers" (under-estimates the jump); too large and bad correspondences leak in.
  double huber_delta = 15.0;
  int irls_iters = 10;
  // Minimum number of (ref,cur) pairs needed to update; otherwise the last parameters are held.
  int min_samples = 15;
  // No-data handling: after hold_frames consecutive frames without an update, pull (g,b) toward identity by 'decay' per frame
  // (stale parameters from a different AGC state are worse than none). Isolated empty frames do not trigger this.
  int hold_frames = 10;
  double decay = 0.05;
  // Temporal smoothing: new = (1-ema_new)*old + ema_new*fit. 1.0 = no smoothing. AGC jumps are abrupt, so keep this high.
  double ema_new = 0.7;
  // Physical plausibility bounds.
  double g_min = 0.6, g_max = 1.6;
  double b_min = -40.0, b_max = 40.0;
  // Frame-level reliability: if the mean NCC of this frame's candidates is below this, the correspondences are mostly wrong
  // (pose/map mismatch) and fitting them only produces a degenerate 'flatten the image' solution -> hold instead.
  double min_mean_ncc = 0.35;
  // Sample gating (used by the caller in vio.cpp).
  // NCC is invariant to gain and bias, so it validates the geometric correspondence independently of the photometry.
  double ncc_thre = 0.5;
  double max_zero_frac = 0.1;  // reject warped ref patches that fall (partly) outside the reference image (they are zero-filled)
};

class PhotometricCalib
{
public:
  PhotometricCalibParams p;

  // Current calibration (maps raw current-frame intensity into the reference photometric frame).
  double g = 1.0;
  double b = 0.0;

  // Diagnostics of the last solve() call (mean absolute patch-mean residual, gray levels).
  int n_used = 0;
  double mae_raw = -1.0;  // with g=1, b=0   (what FAST-LIVO2 sees without calibration)
  double mae_cal = -1.0;  // with the fitted (g,b)

  void reset()
  {
    g = 1.0;
    b = 0.0;
    n_used = 0;
    miss_ = 0;
    mae_raw = mae_cal = -1.0;
    clearSamples();
  }

  // Gate statistics of the current frame (filled by the caller in vio.cpp), to see WHY samples are missing.
  int n_cand = 0;      // map points that reached the sample gate
  int n_rej_zero = 0;  // rejected: warped reference patch partly outside the reference image
  int n_rej_ncc = 0;   // rejected: NCC(warped ref, current) < ncc_thre
  double ncc_sum = 0;  // sum of POST-update NCC over the evaluated candidates
  int n_post = 0;      // number of candidates evaluated after the VIO update
  double ncc_pre_sum = 0;  // same before the VIO update (log only)
  int n_pre = 0;
  // Diagnostic: best NCC within +-search px of the projection (subset of candidates) and where it is found.
  int n_best = 0;
  double d_pos_mm = 0, d_rot_deg = 0;  // size of the pose correction made by the VIO update of this frame
  double ncc_best_sum = 0, dx_sum = 0, dy_sum = 0, adx_sum = 0, ady_sum = 0;
  double img_mean = -1, img_std = -1;  // raw image statistics of this frame (set by the caller)
  int n_pg = 0;        // LiDAR points of this frame given to retrieveFromVisualSparseMap
  int n_vis = 0;       // visible visual-map grid cells (points that could become candidates)

  void clearSamples()
  {
    ref_.clear();
    cur_.clear();
    n_cand = n_rej_zero = n_rej_ncc = n_pg = n_vis = n_post = n_pre = n_best = 0;
    ncc_sum = ncc_pre_sum = ncc_best_sum = dx_sum = dy_sum = adx_sum = ady_sum = 0;
  }

  void addSample(double ref_mean, double cur_raw_mean)
  {
    ref_.push_back(ref_mean);
    cur_.push_back(cur_raw_mean);
  }

  size_t numSamples() const { return ref_.size(); }

  // Robust (Huber/IRLS) fit of (g,b). Returns true if the parameters were updated.
  bool solve()
  {
    const int N = static_cast<int>(ref_.size());
    n_used = N;
    mae_raw = mae_cal = -1.0;
    if (N < p.min_samples)
    {
      if (++miss_ > p.hold_frames)
      {
        g += p.decay * (1.0 - g);
        b *= (1.0 - p.decay);
      }
      return false;
    }
    miss_ = 0;

    // Symmetric robust estimator (replaces least squares of ref on cur).
    // Why: least squares assumes the current-patch means are exact. They are not (pose / warp / depth errors), and then the slope is
    // biased toward 0 ("regression dilution"): wrong correspondences make the fit flatten the image (g -> small, b -> large),
    // which also lowers the MAE, so MAE alone cannot be trusted. The ratio of robust spreads is symmetric in ref and cur.
    //   g = sqrt( (var_ref + s0) / (var_cur + s0) )     s0 = lambda1 (gray^2): shrinks g to 1 when the intensity spread is small
    //   b = median(ref) - g * median(cur)               (shrunk toward 0 for small N)
    if (n_post > 0 && ncc_sum / n_post < p.min_mean_ncc)
    {
      ++miss_;
      return false;  // correspondences of this frame are mostly wrong: do not fit
    }
    auto med = [](std::vector<double> v) {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    auto robust_var = [&](const std::vector<double> &v, double m) {
      std::vector<double> dev(v.size());
      for (size_t k = 0; k < v.size(); k++) dev[k] = std::fabs(v[k] - m);
      const double s = 1.4826 * med(dev);
      return s * s;
    };
    const double m_ref = med(ref_), m_cur = med(cur_);
    const double v_ref = robust_var(ref_, m_ref), v_cur = robust_var(cur_, m_cur);
    double gg = std::sqrt((v_ref + p.lambda1) / (v_cur + p.lambda1));
    double bb = (m_ref - gg * m_cur) * (static_cast<double>(N) / (N + 10.0));
    gg = std::min(std::max(gg, p.g_min), p.g_max);
    bb = std::min(std::max(bb, p.b_min), p.b_max);

    g = (1.0 - p.ema_new) * g + p.ema_new * gg;
    b = (1.0 - p.ema_new) * b + p.ema_new * bb;

    double s_raw = 0, s_cal = 0;
    for (int i = 0; i < N; i++)
    {
      s_raw += std::fabs(ref_[i] - cur_[i]);
      s_cal += std::fabs(ref_[i] - (g * cur_[i] + b));
    }
    mae_raw = s_raw / N;
    mae_cal = s_cal / N;
    return true;
  }

  static void buildLUT(uint8_t lut[256], double gain, double bias)
  {
    for (int i = 0; i < 256; i++)
    {
      const double v = gain * i + bias;
      lut[i] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, std::floor(v + 0.5))));
    }
  }

  // ── CSV log, for plotting g/b over time and for A/B comparisons (mae_raw vs mae_cal) ──
  void openLog(const std::string &path)
  {
    log_tried_ = true;
    log_.open(path, std::ios::out);
    if (!log_.is_open())
    {
      std::fprintf(stderr, "[ photometric_calib ] could not open log file %s\n", path.c_str());
      return;
    }
    log_ << "frame_id,stamp,img_mean,img_std,n_pg,n_vis,n_cand,rej_zero,rej_ncc,mean_ncc_pre,mean_ncc_post,mean_ncc_best,mean_dx,mean_dy,mean_adx,mean_ady,vio_dpos_mm,vio_drot_deg,n_samples,updated,g,b,mae_raw,mae_cal,state_inv_expo\n";
  }
  bool logTried() const { return log_tried_; }

  void logFrame(int frame_id, double stamp, double state_inv_expo, bool updated)
  {
    if (!log_.is_open()) return;
    log_.precision(9);
    log_ << frame_id << "," << stamp << "," << img_mean << "," << img_std << "," << n_pg << "," << n_vis << "," << n_cand << "," << n_rej_zero << "," << n_rej_ncc << ","
         << (n_pre > 0 ? ncc_pre_sum / n_pre : -1.0) << "," << (n_post > 0 ? ncc_sum / n_post : -1.0) << ","
         << (n_best > 0 ? ncc_best_sum / n_best : -1.0) << "," << (n_best > 0 ? dx_sum / n_best : 0.0) << "," << (n_best > 0 ? dy_sum / n_best : 0.0) << ","
         << (n_best > 0 ? adx_sum / n_best : 0.0) << "," << (n_best > 0 ? ady_sum / n_best : 0.0) << "," << d_pos_mm << "," << d_rot_deg << "," << n_used << "," << (updated ? 1 : 0) << "," << g << "," << b << "," << mae_raw << ","
         << mae_cal << "," << state_inv_expo << "\n";
    log_.flush();
  }

private:
  std::vector<double> ref_, cur_;
  std::ofstream log_;
  bool log_tried_ = false;
  int miss_ = 0;
};