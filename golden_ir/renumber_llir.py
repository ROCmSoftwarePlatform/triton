import re

filename = "/home/dtanner/repos/rocm_triton/golden_ir/versions/2_global_loads.llir"
with open(filename, 'r') as f:
  llvm_lines = f.readlines()
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
        pattern_bb = r"\d+:"
        # BB
        if re.match(pattern_bb, token):
            number = int(token[:-1])
            print("BB %i found" % number)
            unv_last = number
        # %123
        elif re.match(pattern_unv, token):
            unv = int(token[1:])
            if unv < min_unv_check:
                # ignore
                unv_last = unv
                continue
            if unv == unv_last+1:
                # valid
                print("%i -> %i" % (unv_last, unv))
                unv_last = unv
            else:
                print("error %i != %i+1; unv_map %i -> %i" % (unv, unv_last, unv, unv_last+1))
                print(number)
                # need to do search-replace for rest of tile
                # change unv to unv_last+1
                unv_map[unv] = unv_last+1
                unv_last += 1

# fix numbering
print(unv_map)
#print(llvm_str)
unique = "__tmp__"
for key, value in unv_map.items():
    key = "%" + str(key)
    value = "%" + unique + str(value)
    llvm_str = llvm_str.replace(key, value)

llvm_str = llvm_str.replace(unique, "")
#print(llvm_str)

# Write.
suffix = ".renumbered"
with open(filename+suffix, 'w') as f:
    f.write(llvm_str)
    f.close()
