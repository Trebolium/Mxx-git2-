import os 
  
# Function to rename multiple files 
def main(): 
      
    for i, file in enumerate(os.listdir("jamendo/labels/")):
        file_name = os.path.basename(file)
        print(file_name)  
        # rename() function will 
        # rename all the files 
        os.rename(file_name, str(i)+file_name) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 