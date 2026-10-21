#Author: Daniel Abadjiev
#Date: June 9, 2026
#Description: script to just train the asic model and then save results (which auto-happens with the .runAllStuff method)


import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import ASICModel


NUM_EPOCHS = 150
tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"

model1 = ASICModel.ModelASIC(tfRecordFolder=tfRecordFolder)
model1.runAllStuff(numEpochs=NUM_EPOCHS,runUnquantized = False)