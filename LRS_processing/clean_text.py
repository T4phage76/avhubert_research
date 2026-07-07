import os
input_dir = 'LRS_processing/LRS2/LRS_main_transcripts'  # folder with original .txt files
output_dir = 'LRS_processing/LRS2/MFA_input/txt'  # folder to save cleaned .txt files

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.txt'):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)
            
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            # Assume first line starts with 'Text: '
            if lines:
                first_line = lines[0].replace('Text: ', '').strip()
            else:
                first_line = ""

            with open(output_path, 'w') as f:
                f.write(first_line + '\n')  # only write the sentence
            
            print(f"Cleaned {file}")