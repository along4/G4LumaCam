# 4. File Structure Explanation
## Overview

The G4LumaCam workflow relies on a structured archive folder to store simulation and analysis data. Users must define this archive directory when initializing simulations or analyses.
## Directory Structure

Below is an example tree structure for the archive/openbeam directory:


    archive/openbeam:
    ├── SimPhotons
    │   ├── macro.mac
    │   ├── sim_data_0.csv
    │   ├── sim_data_1.csv
    │   ├── sim_data_10.csv
    │   ├── sim_data_2.csv
    │   ├── sim_data_3.csv
    │   ├── sim_data_4.csv
    │   ├── sim_data_5.csv
    │   ├── sim_data_6.csv
    │   ├── sim_data_7.csv
    │   ├── sim_data_8.csv
    │   └── sim_data_9.csv
    ├── TracedPhotons
    │   ├── traced_sim_data_0.csv
    │   ├── traced_sim_data_1.csv
    │   ├── traced_sim_data_10.csv
    │   ├── traced_sim_data_2.csv
    │   ├── traced_sim_data_3.csv
    │   ├── traced_sim_data_4.csv
    │   ├── traced_sim_data_5.csv
    │   ├── traced_sim_data_6.csv
    │   ├── traced_sim_data_7.csv
    │   ├── traced_sim_data_8.csv
    │   └── traced_sim_data_9.csv
    ├── PhotonFiles
    │   ├── traced_sim_data_0.empirphot
    │   ├── traced_sim_data_1.empirphot
    │   ├── traced_sim_data_2.empirphot
    │   ├── traced_sim_data_3.empirphot
    │   ├── traced_sim_data_4.empirphot
    │   ├── traced_sim_data_5.empirphot
    │   ├── traced_sim_data_6.empirphot
    │   ├── traced_sim_data_7.empirphot
    │   ├── traced_sim_data_8.empirphot
    │   └── traced_sim_data_9.empirphot
    ├── EventFiles
    │   ├── traced_sim_data_0.empirevent
    │   ├── traced_sim_data_1.empirevent
    │   ├── traced_sim_data_2.empirevent
    │   ├── traced_sim_data_3.empirevent
    │   ├── traced_sim_data_4.empirevent
    │   ├── traced_sim_data_5.empirevent
    │   ├── traced_sim_data_6.empirevent
    │   ├── traced_sim_data_7.empirevent
    │   ├── traced_sim_data_8.empirevent
    │   └── traced_sim_data_9.empirevent
    ├── EventResults
    │   └── batch_0_results.csv
    ├── all_batches_results.csv
    ├── binned.empirevent
    ├── counts.csv
    ├── event_by_event_results.csv
    ├── events_with_shape_parameters.csv
    ├── parameterEvents.json
    └── parameterSettings.json

- SimPhotons: Contains raw simulation data (sim_data_*.csv) and configuration files (macro.mac).
- TracedPhotons: Stores traced photon data after lens processing (traced_sim_data_*.csv).
- EventFiles and PhotonFiles: Hold event and photon data in custom empir formats (.empirevent, .empirphot).
- EventResults: Contains analysis results (e.g., batch_0_results.csv).