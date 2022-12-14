# IT3915 Master preparatory project
## About
Overall problem description (norwegain):

>Gjenfinning av sau ved hjelp av drone \
Gjenfinning av de siste sauene på høsten er ofte en av en sauebondes større mareritt. Han kan ofte legge ned flere hundre timer i felten uten a finne de. Dette er en oppgave som han meget gjerne kan tenke seg teknologisk støtte til å utføre. \
Vi foreslår anvendelse av drone utstyrt med normalt kamera og infrarødt kamera til denne typen søkejobb. Dronen vil fly et veldefinert søkemønster. Kameraene tar bilder som så analyseres på dronen for om det befinner seg sau i bildet. Man vil se sau dels basert på farge og dels basert på at de er vesentlig varmere enn terrenget. \
Vi ønsker her en gruppe på to eller tre studenter. Den ene studenten vil arbeide med utviklingen av bildeanalysen. Den andre vil utføre sammenstilling av de to typer av bilder der det i minst ett av de er detektert mulig sau. Den tredje studenten vil utvikle presentasjonssystemet overfor bønder som viser kartmessig hvor i terrenget sau er funnet. \
Vi har her behov for to studenter som har kunnskap i bildeanalyse og bilde samstilling. Den tredje studenten har generell informasjonsteknologi-bakgrunn. \

The project has been done under the supervision of [Svein-Olaf Hvasshovd](https://www.ntnu.no/ansatte/sophus) at [NTNU](https://www.ntnu.no/), and has focused on image augmentation transformations in order to improve the original dataset. This is a project meant to prepare for the master's thesis.

This implementation uses the state of the art machine learning algorithm for object detection - [YOLOv7](https://github.com/WongKinYiu/yolov7), with [this dataset](https://app.roboflow.com/it3915masterpreparatoryproject/merged-sheep-dataset/3).


## Installation

- [Connect to the IDUN GPU cluster via SSH](https://www.hpc.ntnu.no/idun/getting-started-on-idun/login/). This requires the user to be connected to the NTNU intranet either by being directly connected to the "eduroam" wifi on campus, or by using a [VPN](https://i.ntnu.no/wiki/-/wiki/norsk/installere+vpn).
- Clone [this repository](https://github.com/trymgrande/IT3915-master-preparatory-project) on the login node after connecting.

## Running on the NTNU Idun Cluster
See the full documentation for [running jobs on the IDUN GPU Cluster](https://www.hpc.ntnu.no/idun/getting-started-on-idun/running-jobs/).
A job can be queued by running:
```bash
sbatch job.slurm <augmentation_id>
```
where the "augmentation_id" is a number corresponding to the desired augmentation transformation found in [augmentation_transformations.py](augmentation_transformations.py). Each job will automatically be assigned a unique job ID by the GPU cluster.

The job-environment-<job_id>-<augmentation_id> directory contains all the necessary files for performing augmentation, training, and testing on a new copy of the dataset. The training results are copied from this directory, into runs/run-<job_id>-<augmentation_id>. A log for each job is generated with the name "slurm-<job_id>, where the entire pipeline process is logged including errors if any occur. After the results are verified and the job has been completed successfully, the job-environment directory may be deleted.

## Report
See [Project Report](Project_Report.pdf).