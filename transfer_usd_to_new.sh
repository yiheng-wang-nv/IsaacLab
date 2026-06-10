# 2. healthcare_assets (158M) - 包含 robot/scene/tray
rsync -avh --progress \
  /localhome/local-vennw/models/healthcare_assets/ \
  nvidia@172.29.225.93:/home/nvidia/workspace/yiheng/assets/healthcare_assets/

# 3. sim2real_rlinf_dev_pg assets (19M) - trocar_1 nodeform
rsync -avh --progress \
  /localhome/local-vennw/code/sim2real_rlinf_dev_pg/assets/ \
  nvidia@172.29.225.93:/home/nvidia/workspace/yiheng/assets/sim2real_rlinf_dev_pg/assets/

# 4. trocar_2 (2.4M) - 单个文件
scp /localhome/local-vennw/code/DisposableLaparoscopicPunctureDevice006_test.usd \
  nvidia@172.29.225.93:/home/nvidia/workspace/yiheng/assets/
