import jmptools

rval, rmessage, df = jmptools.readjmp("TestFile.jmp")
print df
print rval, rmessage
# write to csv if you want 
#jmptools.to_csv("TestFile.csv")