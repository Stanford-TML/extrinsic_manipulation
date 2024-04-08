import subprocess
import pickle
import os
import json
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_examples', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, default='contactdemo/data/pushing/demo')
    parser.add_argument('--n_process', type=int, default=16)

    args = parser.parse_args()
    n_process = args.n_process
    n_examples = args.n_examples
    dataset_path = args.dataset_path

    n_examples_each = n_examples // n_process
    start_idxs = [i * n_examples_each for i in range(n_process)]

    ps = []
    for i in range(n_process):
        if i == n_process - 1:
            n_examples_each = n_examples - start_idxs[i]
        start_idx = start_idxs[i]
        command = [
            'python',
            'contactdemo/lib/data_generation/general_pushing_data_generation_scanned.py',
            f'--start_idx={start_idx}',
            f'--n_examples={n_examples_each}',
            f'--seed={i}',
            f'--dataset_path={dataset_path}',
        ]
        print(command)
        p = subprocess.Popen(command)
        ps.append(p)
    

    try:
        while ps:
            for p in ps:
                if p.poll() is not None:  # Process has finished
                    ps.remove(p)
                    if p.returncode != 0:  # Process ended with an error
                        print(f"Process {p.args} exited with error code {p.returncode}. Terminating others.")
                        # Terminate all other processes
                        for other_p in ps:
                            other_p.terminate()
                        ps = []  # Clear the list to break the outer loop
                    else:
                        print(f"Process {p.args} completed successfully.")
            time.sleep(0.5)  # Avoid busy waiting
    except KeyboardInterrupt:
        # Handle cases where the script is interrupted
        for p in ps:
            p.terminate()
        print("Terminated all processes due to keyboard interrupt.")

    dataset = {}
    for i in range(n_process):
        if i == n_process - 1:
            n_examples_each = n_examples - start_idxs[i]
        else:
            n_examples_each = n_examples // n_process
        
        with open(os.path.join(dataset_path, f"{start_idxs[i]}_{start_idxs[i]+n_examples_each}.json"), "r") as fin:
            data_list = json.load(fin)
        
        for data in data_list:
            idx = data['obj_name']
            if idx in dataset:
                dataset[idx].append(data)
            else:
                dataset[idx] = [data]

    with open(dataset_path + '.pkl', "wb") as fout:
        pickle.dump(dataset, fout)
    
    print('dataset saved to', dataset_path + '.pkl')
    with open(dataset_path + '.pkl', "rb") as fin:
        dataset = pickle.load(fin)

