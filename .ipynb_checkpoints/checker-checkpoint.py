#read from the input file
inputFile = open('SMSSpamCollection.txt','r')
#write to the pass file and failFile
passFile = open('ham.txt','w')
failFile = open('spam.txt','w')

for line in inputFile:
    line_split = line.split()
    if line_split[0].lower()=='ham':
        ham.write(line)
    else: spam.write(line)


inputFile.close()
ham.close()
spam.close()