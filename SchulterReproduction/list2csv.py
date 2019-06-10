	prediction_list.append(frame_index_Value)

with open(sys.argv[2], "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(prediction_list)
csvFile.close()