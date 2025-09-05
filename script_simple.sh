


# LearnerValueSystemLearningPolicy
# Default values
env="ffmo"
mode="all"
expol="envelope"
pol="envelope"
algo="pc"
L=2
seed=26
#seeds_list=(26 27 28 29 30)
sname="default"
skip_confirm=false
wandb=true
prefix=""
prefix_data=""
resume_from=0
# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    -trval)
            mode="traineval"
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
        -rw)
            env="rw"
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
        -prefix)
            prefix="$2"
            shift 2
            ;;
        -exenve)
            expol="envelope"
            shift
            ;;
        -lenve)
            pol="envelope"
            shift
            ;;
        -y)
            skip_confirm=true
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
            L="$2"
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
        -algo)
            algo="$2"
            shift 2
            ;;
        -seed)
            seed="$2"
            shift 2
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


echo "Running with: "
echo "ENVIRONMENT": $env
echo "MODE": $mode
echo "EXPERT POLICY": $expol
echo "LEARNER POLICY": $pol
echo "ALGORITHM": $algo
echo "Number of clusters L": $L
echo "SEEDS": $seed
echo "SKIP CONFIRMATION": $skip_confirm
echo "RESUME FROM": $resume_from
echo "WANDB": $wandb
echo "PREFIX=": $prefix
if [[ "$env" != "ffmo" && "$env" != "mvc" && "$env" != "rw" && "$env" != "dst" && "$env" != "mine" ]]; then
    echo "Error: env must be one of: ffmo, mvc, rw, dst, mine"
    exit 1
fi
if [[ "$algo" != "pc" && "$algo" != "pbmorl" && "$algo" != "free" && "$algo" != "cpbmorl"  ]]; then
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

echo "NEW LEARNER POLICY": $pol

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
    python generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -rt -pareto -a $algo -cf algorithm_config_${algo}.json
    python generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -pareto -a $algo -cf algorithm_config_${algo}.json
elif [[ "$mode" == "all" || "$mode" == "gen" ]]; then
    echo "Generating dataset for $env with expol $expol and algorithm $algo"
    python generate_dataset_mo.py -e $env -sname $sname -dname ${prefix_data}mo_${env}_${expol} -expol $expol -gentr -genpf -ts=0.5 -pareto -a $algo -cf algorithm_config_${algo}.json
fi
if [[ "$mode" == "all" || "$mode" == "train" || "$mode" == "traineval" ]]; then
    if [[ $wandb == true ]]; then
        python train_movsl.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol}_L${L}_seed${seed} -sname $sname  -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -e $env -L $L -a $algo -s $seed -wb -rs $resume_from -cf algorithm_config_${algo}.json
    else 
        python train_movsl.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol}_L${L}_seed${seed} -sname $sname  -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -e $env -L $L -a $algo -s $seed -rs $resume_from -cf algorithm_config_${algo}.json
    fi
fi
echo "Training completed."
if [[ "$mode" == "all" || "$mode" == "eval" || "$mode" == "traineval" ]]; then
    python evaluate.py -ename ${prefix}mo_${algo}_${env}_${pol}_from_${expol}_L${L}_seed${seed} -sname $sname -expol $expol -pol $pol -dname ${prefix_data}mo_${env}_${expol} -e $env -strajs 200 -seps 0.05 -a $algo -cf algorithm_config_${algo}.json
fi
