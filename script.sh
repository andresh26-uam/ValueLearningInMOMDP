


# LearnerValueSystemLearningPolicy
# Default values
env="ffmo"
mode="all"
L=(1 4 10)
expol="envelope"
pol="envelope"
algo="pc"
seeds_list=(26 27 28 29 30)
sname="default"
wandb=true
skip_confirm=false
resume_from=0
prefix=""
prefix_data="useprefix"
O=""
# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -trval)
            mode="traineval"
            shift
            ;;
    -O)
            O="-O"
            shift 
            ;;
    -all)
            mode="all"
            shift
            ;;
        -train)
            mode="train"
            shift
            ;;
        -eval)
            mode="eval"
            shift
            ;;
        -genrt)
            mode="genrt"
            shift
            ;;
        -gen)
            mode="gen"
            shift
            ;;
        -ffmo)
            env="ffmo"
            shift
            ;;
        -ff)
            env="ffmo"
            shift
            ;;
        -mine)
            env="mine"
            shift
            ;;
        -mvc)
            env="mvc"
            shift
            ;;
        -dst)
            env="dst"
            shift
            ;;
        -exppo)
            expol="ppo_learner"
            shift
            ;;
        -lppo)
            pol="ppo_learner"
            shift
            ;;
        -exenve)
            expol="envelope"
            shift
            ;;
        -pdata)
            prefix_data="$2"
            shift 2
            ;;
        -lenve)
            pol="envelope"
            shift
            ;;
        -env)
            env="$2"
            shift 2
            ;;
        -mode)
            mode="$2"
            shift 2
            ;;
        -L)
            IFS=',' read -ra L <<< "$2"
            shift 2
            ;;
        -expol)
            expol="$2"
            shift 2
            ;;
        -pol)
            pol="$2"
            shift 2
            ;;
        -sname)
            sname="$2"
            shift 2
            ;;
        -algo)
            algo="$2"
            shift 2
            ;;
        -seeds)
            IFS=',' read -ra seeds_list <<< "$2"
            shift 2
            ;;
        -prefix)
            prefix="$2"
            shift 2
            ;;
        -y|--yes)
            skip_confirm=true
            shift
            ;;
        -rw)
            env="rw"
            shift
            ;;
        -rs)
            resume_from="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ "$prefix_data" == "useprefix" ]]; then
    prefix_data=$prefix
fi
echo "Running with: "
echo "ENVIRONMENT": $env
echo "MODE": $mode
echo "EXPERT POLICY": $expol
echo "LEARNER POLICY": $pol
echo "ALGORITHM": $algo
echo "Number of clusters L": ${L[@]}
echo "SEEDS": ${seeds_list[@]}
echo "WANDB": $wandb
echo "RESUME FROM": $resume_from
echo "DATA PREFIX": $prefix_data
if [[ "$env" != "ffmo" && "$env" != "mvc" && "$env" != "rw" && "$env" != "dst" && "$env" != "mine" ]]; then
    echo "Error: env must be one of: ffmo, mvc, dst, mine"
    exit 1
fi
if [[ "$algo" != "pc" && "$algo" != "pbmorl" && "$algo" != "free" && "$algo" != "cpbmorl" ]]; then
    echo "Error: algo must be one of: pbmorl, pc, free, cpbmorl"
    exit 1
fi

if [[ "$mode" != "all" && "$mode" != "gen" && "$mode" != "genrt" && "$mode" != "train"  && "$mode" != "eval" && "$mode" != "traineval" ]]; then
    echo "Error: mode must be one of: all, gen, genrt, train, eval, traineval"
    exit 1
fi
if [[ "$pol" != "ppo_learner" && "$pol" != "envelope" ]]; then
    echo "Error: policy must be one of: ppo_learner, envelope"
    exit 1
fi
if [[ "$pol" == "envelope" ]]; then
    pol="EnvelopeCustomReward"
fi

if [[ "$algo" == "pbmorl" ]]; then
    pol="EnvelopePBMORL"
fi
if [[ "$algo" == "cpbmorl" ]]; then
    pol="EnvelopeClusteredPBMORL"
fi
if [[ "$expol" != "ppo_learner" && "$expol" != "envelope" ]]; then
    echo "Error: policy must be one of: ppo_learner, envelope"
    exit 1
fi
if ! $skip_confirm; then
    read -p "Proceed with the above configuration? (y/n): " confirm || { echo "No input received. Aborted."; exit 1; }
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborted by user."
        exit 1
    fi
fi


if [[ "$mode" == "all" || "$mode" == "genrt" ]]; then
    echo "Generating dataset for $env with expol $expol and algorithm $algo RETRAINING!"
    python $O generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -rt -pareto -a $algo -cf algorithm_config_${algo}.json
    python $O generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -pareto -a $algo -cf algorithm_config_${algo}.json

elif [[ "$mode" == "all" || "$mode" == "gen" ]]; then
    echo "Generating dataset for $env with expol $expol and algorithm $algo"
    python $O generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -pareto -a $algo -cf algorithm_config_${algo}.json
fi
if [[ "$mode" == "all" || "$mode" == "train" || "$mode" == "traineval" ]]; then
    
    pids=()
    for l in "${L[@]}"; do
        for seed in "${seeds_list[@]}"; do
            if [[ $wandb == true ]]; then
                python $O train_movsl.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol}_L${l}_seed${seed} -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -sname $sname -e $env -L $l -a $algo -s $seed -rs $resume_from -cf algorithm_config_${algo}.json &
                pids+=($!)
            else
                python $O train_movsl.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol}_L${l}_seed${seed} -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -sname $sname -e $env -L $l -a $algo -s $seed -rs $resume_from -cf algorithm_config_${algo}.json &
                pids+=($!)
            fi
            
        done
    done

    # Wait for all jobs and check exit codes
    for pid in "${pids[@]}"; do
        wait $pid
        status=$?
        if [[ $status -ne 0 ]]; then
            echo "A training process (PID $pid) failed with exit code $status. Aborting script."
            exit 1
        fi
    done
    echo "Training completed."
fi

if [[ "$mode" == "all" || "$mode" == "eval" || "$mode" == "traineval" ]]; then
    seeds_str=$(IFS=,; echo "${seeds_list[*]}")
    Lstr=$(IFS=,; echo "${L[*]}")
    python $O evaluate.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol} -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -sname $sname -e $env -strajs 200 -seps 0.05 -a $algo -seeds $seeds_str -Ltries $Lstr -cf algorithm_config_${algo}.json
fi