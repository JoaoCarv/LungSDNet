import sys
import time as t
curret = t.time()
module_path = r'C:\JFCImportantes\Universidade\Thesis\DeepLung\my_utilities'

sys.path.insert(0, module_path)
# import utilities
import sitk_ops

# from my_utilities import utilities as utl

print(t.time()-curret)
help(utilities.beep_sound)
#utilities.beep_sound()
