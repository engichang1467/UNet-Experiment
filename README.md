# UNet Experiment

This project uses UNet to experiment with detecting forecast turbulence.

## How to Run It

Install the necesary libraries
```sh
pip install -r requirements.txt
```

### Generate Data

1. Download the dataset file `snapshots_s1.h5`.
2. Create a folder named `data` and place the dataset file into this folder.
3. Run the following command to generate the data:
    ```sh
    sh data_gen.sh
    ```

### Train the Model

Run the following command to train the model:
```sh
sh run_experiment.sh
```

## References

- [maxjiang93: MeshfreeFlowNet](https://github.com/maxjiang93/space_time_pde)