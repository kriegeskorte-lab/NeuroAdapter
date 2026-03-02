#!/bin/bash
for start in $(seq 0 20 500); do
    end=$((start + 19))        # end of range (start + 19)
    if [ $end -gt 514 ]; then  # cap at 514
        end=514
    fi

    idx=""
    for i in $(seq $start $end); do
        idx="$idx $i"          # build index string: "0 1 2 ... 19"
    done

    echo "Running with --selected_idx $idx"

    python extract_brain_adapter.py \
        --model_weights_dir brain_adapter/model_weights/09_20_2025-21_34 \
        --saved_epochs 300 \
        --selected_idx $idx \
        --subject_id 1 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection \
        --save_frequency 1
done

# for start in $(seq 0 20 500); do
#     end=$((start + 19))        # end of range (start + 19)
#     if [ $end -gt 514 ]; then  # cap at 514
#         end=514
#     fi

#     idx=""
#     for i in $(seq $start $end); do
#         idx="$idx $i"          # build index string: "0 1 2 ... 19"
#     done

#     echo "Running with --selected_idx $idx"

#     python extract_brain_adapter.py \
#         --model_weights_dir brain_adapter/model_weights/09_12_2025-10_41 \
#         --saved_epochs 200 \
#         --selected_idx $idx \
#         --subject_id 2 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection \
#         --save_frequency 1
# done
