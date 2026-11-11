# Cross-illumination boundary test

## Role in the manuscript

E8: zero-shot deployment of the nominal-scene model into a photometrically
shifted world, without recalibration.

## Shifted world

- Dim warm light, darkened sky, and heavier fog.
- Same geometry, class definitions, and kinematic trajectory as the nominal
  world.
- No model retraining, no scaler refitting, and no threshold adjustment.

## Observed boundary

- Visual health drops to mean s_v = 0.324 and the visual weight to
  w_v = 0.203, automatically shifting reliance to depth.
- Online semantic accuracy falls from 90.1% to 73.3%.
- SAPF 3D RMSE rises from 0.088 m to 0.149 m.
- Flat recall is suppressed by fog-induced contrast loss.

This confirms that health gating cushions severe photometric shift but cannot
replace target-domain calibration or online illumination normalization.

## Uploaded files

- `figures/cross_illumination/predictions.npz`
- `figures/cross_illumination/sapf_gazebo_result.json`
- `figures/cross_illumination/sapf_gazebo_trajectory.npz`
- `figures/cross_illumination/perception.log`
- `figures/cross_illumination/trajectory_xz.png`
- `figures/cross_illumination/semantic_labels.png`
- `ros_ws/run_cross_illumination_go1.sh`,
  `ros_ws/worlds/guide_dog_street_shifted.world`
