import re
import sys
#import os
#pwd = os.path.dirname(__file__)

# filename = "/home/dtanner/repos/rocm_triton/golden_ir/versions/2_global_loads.llir"
filename = sys.argv[1]
print(filename)
with open(filename, 'r') as f:
  llvm_lines = f.readlines()
  print(len(llvm_lines))
  llvm_str = "".join(llvm_lines)
  f.close()

# unnamed values
unv_last = -1
min_unv_check = 100
num_lines = len(llvm_lines)
#num_lines = 1000
unv_map = {}
for ii in range(0, num_lines):
    line = llvm_lines[ii]
    s = line.split()
    #print(s)
    if len(s) > 0:
        token = s[0]
        pattern_unv = r"\%\d+"
        pattern_bb = r"^\d+:"
        # BB
        if re.match(pattern_bb, token):
            bbid = int(token[:-1])
            print("Found BB: %i -> %i" % (bbid, unv_last))
            unv_map[bbid] = unv_last+1
            unv_last += 1
        # %123
        elif re.match(pattern_unv, token):
            unv = int(token[1:])
            if unv < min_unv_check:
                # ignore
                unv_last = unv
                continue
            if unv == unv_last+1:
                # valid
                #print("%i -> %i" % (unv_last, unv))
                unv_last = unv
            else:
                print("Found: %i -> %i" % (unv, unv_last+1))
                # need to do search-replace for rest of tile
                # change unv to unv_last+1
                unv_map[unv] = unv_last+1
                unv_last += 1

# fix numbering
#print(unv_map)

#print(llvm_str)
unique = "__tmp__"
for key, value in unv_map.items():
    # update unv
    for suffix in [' ', ',']:
        sub_key = "%" + str(key) + suffix
        sub_value = "%" + unique + str(value) + suffix
        llvm_str = llvm_str.replace(sub_key, sub_value)
    # update bb
    #key = '\n' + str(key) + ':'
    #value = '\n' + unique + str(value) + ':'
    sub_key = '\n' + str(key) + ':'
    sub_value = '\n' + unique + str(value) + ':'
    #print("Key: {%s}" % sub_key)
    #print("Value: {%s}" % sub_value)
    llvm_str = llvm_str.replace(sub_key, sub_value)

llvm_str = llvm_str.replace(unique, "")
#print(llvm_str)

# Write.
suffix = ".renumbered"
with open(filename+suffix, 'w') as f:
    f.write(llvm_str)
    f.close()
