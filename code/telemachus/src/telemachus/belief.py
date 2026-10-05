from __future__ import annotations

import numpy as np
from .vision import camera_quality_features


class MapBelief:
    def __init__(self, n, cell_size, pose_georeferencing=True):
        self.cell_size = cell_size
        self.pose_georeferencing = pose_georeferencing
        self.log_odds = np.full((n,n), np.log(.12/.88))
        self.observations = np.zeros((n,n), dtype=int)
        self.best_quality = np.zeros((n,n), dtype=float)
        self.last_entropy = 0.
        self.last_disagreement = 0.
        self.revision = 0.
        self.quality = {"image_high_frequency": 0., "image_contrast": 0., "image_brightness": .8}

    @property
    def probability(self):
        return 1/(1+np.exp(-np.clip(self.log_odds,-7,7)))

    def update(self, reading, vision, pose):
        p, variance, entropy = vision.predict(reading.tiles)
        self.last_entropy = float(np.mean(entropy))
        self.last_disagreement = float(np.mean(variance))
        self.quality = camera_quality_features(reading.tiles)
        reliability=float(np.clip(1-max(0,self.quality["image_high_frequency"]-.08)/.26,.08,1.))
        if len(reading.gps_burst)>1:
            measured_position_variance=float(np.mean(np.var(reading.gps_burst[:,:2],axis=0,ddof=1)))
            # The image may be sharp while its georeferencing is uncertain.
            # Estimate this from burst scatter, never from scenario sigma.
            reliability*=float(np.exp(-max(0,measured_position_variance-.003)/.06))
        # Measurable distribution shift creates an uncertainty floor even
        # when the synthetic CNN is overconfident. Same rule for all methods.
        p=.5+reliability*(p-.5)
        centers=reading.tile_centers
        if self.pose_georeferencing and len(reading.gps_burst):
            # Sensor coordinates were georeferenced with the received GPS.
            # Fuse the same measured frame using the filtered pose, rather
            # than treating the noisy GPS origin as exact. No true pose used.
            centers=centers+np.asarray(pose[:2])-np.mean(reading.gps_burst[:,:2],axis=0)
        self._fuse(centers, p, pose, reading.focus,reliability)

    def update_remote(self, centers, probabilities, variance, pose):
        p = np.clip(probabilities, .001, .999)
        self.last_entropy = float(np.mean(-(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)))
        self.last_disagreement = float(np.mean(variance))
        self._fuse(centers, p, pose, True,1.)

    def _fuse(self, centers, p, pose, focused,reliability):
        rc = np.floor(np.asarray(centers)[:, ::-1]/self.cell_size).astype(int)
        valid = (rc >= 0).all(1) & (rc < self.log_odds.shape[0]).all(1)
        rc = rc[valid]; p = p[valid]
        if len(p):
            previous = self.probability[rc[:,0],rc[:,1]]
            seen = self.observations[rc[:,0],rc[:,1]] > 0
            self.revision = float(np.mean(np.abs(previous[seen]-p[seen]))) if np.any(seen) else 0.
            # Correlated repeated camera views must not generate unbounded
            # confidence. Exponential fusion has a capped evidence weight.
            logits = np.log(p/(1-p))
            old = self.log_odds[rc[:,0],rc[:,1]]
            best=self.best_quality[rc[:,0],rc[:,1]]
            weight=np.where(reliability+.1<best,.04,.90 if focused else .65)
            self.log_odds[rc[:,0],rc[:,1]] = np.clip((1-weight)*old+weight*logits,-6,6)
            self.observations[rc[:,0],rc[:,1]] += 1
            self.best_quality[rc[:,0],rc[:,1]]=np.maximum(best,reliability)
        r,c = np.clip(np.floor(np.asarray(pose[:2])[::-1]/self.cell_size).astype(int),0,len(self.log_odds)-1)
        self.log_odds[r,c] = -6


FEATURE_NAMES = ("camera_entropy", "camera_disagreement", "camera_revision", "image_high_frequency",
                 "image_contrast", "pose_noise", "pose_covariance", "innovation_energy",
                 "dynamics_bias", "dynamics_residual", "gain_deviation", "gain_uncertainty",
                 "channel_failure", "pilot_snr")


def observed_features(map_belief, pose_filter, dynamics, channel):
    m, p, d, c = map_belief, pose_filter, dynamics, channel
    return np.array([m.last_entropy, m.last_disagreement, m.revision,
                     m.quality["image_high_frequency"],m.quality["image_contrast"],
                     p.noise_evidence, np.trace(p.cov[:2,:2]), p.innovation_energy,
                     abs(d.bias_ewma), d.residual_ewma, abs(d.mean-1), d.std,
                     1-c.delivery_probability, c.snr_db], dtype=float)


class ObservableDiagnosis:
    """Interpretable source scores derived exclusively from sensor residuals.

    Scores are resource routing heuristics, not calibrated fault probabilities.
    Multiple sources may be active simultaneously. Training fault names and
    simulator parameters are deliberately absent from this runtime interface.
    """
    def predict(self, features):
        f = dict(zip(FEATURE_NAMES, np.asarray(features, float)))
        vision = max(f["camera_entropy"]/.6, f["camera_revision"]/.4,
                     max(0,f["image_high_frequency"]-.10)/.25)
        localization = max(f["pose_noise"]/.035, f["pose_covariance"]/.025)
        # A learned gain different from 1 is not itself a remaining model
        # error. Residual bias and estimation uncertainty justify new probes.
        model = max(f["dynamics_bias"]/.065, f["gain_uncertainty"]/.22)
        channel = max(f["channel_failure"]/.75, (3-f["pilot_snr"])/8)
        return dict(zip(("vision","localization","model","channel"),
                        np.clip([vision,localization,model,channel],0,1).tolist()))
