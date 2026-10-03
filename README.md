# Learning to control satellite's attitude

## This project present an ADCS simulation and verification project with learned reaction-wheel control


The project evolves, current stage demonstrates a satellite's attitude control subsystem. It does not yet demonstrate docking or manipulation. Principal milestones are described in project_evolution.md file. 


### Train agent from scratch
python -m simulators.fdir.train --guidance-mode ground_station --target-lat-deg 8.49 --target-lon-deg -13.23 --orbit-inclination-deg 10.0 --orbit-raan-deg 288.93 --orbit-argument-latitude-deg 58.23 --ground-pass-reset-mode pass_centered

### Train agent from checkpoint
python -m simulators.fdir.train --guidance-mode ground_station --target-lat-deg 8.49 --target-lon-deg -13.23 --orbit-inclination-deg 10.0 --orbit-raan-deg 288.93 --orbit-argument-latitude-deg 58.23 --ground-pass-reset-mode random_visible --load-from-existing weights\station_tracking_update.pkl

### Evaluate agent
python -m simulators.fdir.evaluate_checkpoint weights\station_tracking_update.pkl --video eval_output/agent_ground_station_performance.mp4 --eval-envs 10