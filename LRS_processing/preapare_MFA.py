import subprocess

subprocess.run([
    "mfa","model","download", "dictionary", "english_us_mfa", 'LRS_processing/MFA_models'
])

subprocess.run([
    "mfa", "download", "acoustic", "english_mfa", "LRS_processing/MFA_models"
])


"""
mfa align 
        LRS_processing/LRS2/MFA_input 
        LRS_processing/LRS2/models/MFA/pretrained_models/dictionary/english_us_mfa.dict 
        LRS_processing/LRS2/models/MFA/pretrained_models/acoustic/english_mfa.zip 
        LRS_processing/LRS2/MFA_output 
        --clean
"""


cmd = [
    "mfa", "align",
    "LRS_processing/LRS2/MFA_input", # input corpus
    "LRS_processing/LRS2/models/MFA/pretrained_models/dictionary/english_us_mfa.dict", # english dictionary 
    "LRS_processing/LRS2/models/MFA/pretrained_models/acoustic/english_mfa.zip", # model
    "LRS_processing/LRS2/MFA_output", # output dir
    "--clean"] # save memory

subprocess.run(cmd, check=True)


