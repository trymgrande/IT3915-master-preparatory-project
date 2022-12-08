# IT3915 Master preparatory project

## Installation

- [Connect to the IDUN GPU cluster via SSH](https://www.hpc.ntnu.no/idun/getting-started-on-idun/login/). This requires the user to be connected to the NTNU intranet either by being directly connected to the "eduroam" wifi on campus, or by using a [VPN](https://i.ntnu.no/wiki/-/wiki/norsk/installere+vpn).
- Clone [this repository](https://github.com/trymgrande/IT3915-master-preparatory-project) on the login node after connecting.

## Running
- See [documentation for running jobs](https://www.hpc.ntnu.no/idun/getting-started-on-idun/running-jobs/) on the IDUN GPU Cluster.
- A job can be run by running:
```bash
sbatch job.slurm <augmentation_id>
```
where the "augmentation_id" is a number corresponding to the desired augmentation transformation found in [augmentation_transformations.py](augmentation_transformations.py).