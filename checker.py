# #read from the input file
# inputFile = open('SMSSpamCollection.txt','r')
# #write to the pass file and failFile
# hamFile = open('ham.txt','w')
# spamFile = open('spam.txt','w')

# for line in inputFile:
#     line_split = line.split()
#     if line_split[0].lower()=='ham':
#         hamFile.write(line)
#     else: spamFile.write(line)

# inputFile.close()
# ham.close()
# spam.close()


# Read from the input file



# inputFile = open('SMSSpamCollection.txt', 'r')
# # Write to the ham file and spam file
# hamFile = open('ham.txt', 'w')
# spamFile = open('spam.txt', 'w')

# for line in inputFile:
#     line_split = line.split()
#     if line_split[0] == 'ham':
#         hamFile.write(line)
#     else:
#         spamFile.write(line)

# # Close all opened files
# inputFile.close()
# hamFile.close()
# spamFile.close()




# Read from the input file
inputFile = open('SMSSpamCollection.txt', 'r')
# Write to the ham file and spam file
hamFile = open('ham.txt', 'w')
spamFile = open('spam.txt', 'w')

for line in inputFile:
    line_split = line.split('\t')  # Split using tab as delimiter
    if len(line_split) >= 2:  # Ensure the line has at least two parts
        label, message = line_split[0], line_split[1]
        if label.lower() == 'ham':
            hamFile.write(message + '\n')
        elif label.lower() == 'spam':
            spamFile.write(message + '\n')

# Close all opened files
inputFile.close()
hamFile.close()
spamFile.close()
