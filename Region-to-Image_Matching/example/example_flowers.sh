if [ "$#" -ne "2" ]; then
  echo "Usage:  ./example.sh <real_image> <gan_image>"
  exit 1
fi


real_image="${1}"
gan_image="${2}"

python example.py ${real_image} ${gan_image}
python extract_train_test_DBH_flower.py data_feature/train.txt data_feature/test.txt 
python image_matching.py data_feature/train_feature.txt data_feature/train_label.txt data_feature/test_feature.txt

