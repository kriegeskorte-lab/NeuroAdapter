
# Define model/subject pairs
declare -A pairs=(
    ["09_20_2025-21_34"]="1"
    ["09_15_2025-12_52"]="2"
    ["09_16_2025-12_06"]="5"
    ["09_20_2025-14_23"]="7"
)

# Generate space-separated img_idx list (0..99)
# img_idx=$(seq -s " " 0 99)

# for model_name in "${!pairs[@]}"; do
#     subject_id=${pairs[$model_name]}
#     echo "Running model=$model_name subject=$subject_id"

#     python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/${model_name} \
#         --saved_epochs 300 \
#         --img_idx $img_idx \
#         --subject_id $subject_id \
#         --num_predictions 7 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection
# done

img_idx=$(seq -s " " 100 199)
for model_name in "${!pairs[@]}"; do
    subject_id=${pairs[$model_name]}
    echo "Running model=$model_name subject=$subject_id"

    python decode_brain_adapte_examples.py \
        --model_weights_dir brain_adapter/model_weights/${model_name} \
        --saved_epochs 300 \
        --img_idx $img_idx \
        --subject_id $subject_id \
        --num_predictions 7 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection
done

img_idx=$(seq -s " " 200 299)
for model_name in "${!pairs[@]}"; do
    subject_id=${pairs[$model_name]}
    echo "Running model=$model_name subject=$subject_id"

    python decode_brain_adapte_examples.py \
        --model_weights_dir brain_adapter/model_weights/${model_name} \
        --saved_epochs 300 \
        --img_idx $img_idx \
        --subject_id $subject_id \
        --num_predictions 7 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection
done

img_idx=$(seq -s " " 300 399)
for model_name in "${!pairs[@]}"; do
    subject_id=${pairs[$model_name]}
    echo "Running model=$model_name subject=$subject_id"

    python decode_brain_adapte_examples.py \
        --model_weights_dir brain_adapter/model_weights/${model_name} \
        --saved_epochs 300 \
        --img_idx $img_idx \
        --subject_id $subject_id \
        --num_predictions 7 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection
done

img_idx=$(seq -s " " 400 514)
for model_name in "${!pairs[@]}"; do
    subject_id=${pairs[$model_name]}
    echo "Running model=$model_name subject=$subject_id"

    python decode_brain_adapte_examples.py \
        --model_weights_dir brain_adapter/model_weights/${model_name} \
        --saved_epochs 300 \
        --img_idx $img_idx \
        --subject_id $subject_id \
        --num_predictions 7 \
        --noise_factor 4.0 \
        --topk 100 \
        --condition_dim 768 \
        --sub_approach linear_projection
done

# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_20_2025-14_23 \
#         --saved_epochs 300 \
#         --img_idx 0  \
#         --subject_id 7 \
#         --num_predictions 8 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection


# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_15_2025-12_52 \
#         --saved_epochs 300 \
#         --img_idx 5 6 11 19 27 38 60 62 0 22 31 50 57 13 69 70 103 81 45 61 75 82 105 95 \
#         --subject_id 2 \
#         --num_predictions 8 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection
# 
# 5 6 11 19 27 38 60 62 0 22 31 50 57 13 69 70 103 81 45 61 75 82 105 95
# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_16_2025-12_06 \
#         --saved_epochs 300 \
#         --img_idx 5 6 11 19 27 38 60 62 0 22 31 50 57 13 69 70 103 81 45 61 75 82 105 95 \
#         --subject_id 5 \
#         --num_predictions 8 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection

# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_20_2025-14_23 \
#         --saved_epochs 300 \
#         --img_idx 5 6 11 19 27 38 60 62 0 22 31 50 57 13 69 70 103 81 45 61 75 82 105 95 \
#         --subject_id 7 \
#         --num_predictions 8 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection

# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_15_2025-12_52 \
#         --saved_epochs 300 \
#         --img_idx 61 75 \
#         --subject_id 2 \
#         --num_predictions 4 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection

# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_16_2025-12_06 \
#         --saved_epochs 300 \
#         --img_idx 81 57 \
#         --subject_id 5 \
#         --num_predictions 4 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection

# python decode_brain_adapte_examples.py \
#         --model_weights_dir brain_adapter/model_weights/09_20_2025-14_23 \
#         --saved_epochs 300 \
#         --img_idx 83 126 253 25 119 351 31 105 423 511 490 484 \
#         --subject_id 7 \
#         --num_predictions 4 \
#         --noise_factor 4.0 \
#         --topk 100 \
#         --condition_dim 768 \
#         --sub_approach linear_projection