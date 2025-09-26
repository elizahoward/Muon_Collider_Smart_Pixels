from Muon_Collider_Smart_Pixels.MuC_Smartpix_Data_Production.Tracklist_Production.tracklist_utils_old import getTracks
import subprocess

# Run set-up script
subprocess.run(['source', 'set_up.sh'], check=True)
print("Set up complete.")
 


plot = False

file_paths = "./output_sim.slcio"
getTracks(file_paths, allowedPIDS=[13], plot=plot, max_events=-1, flp=0, tracklist_folder="Muon_Collider_Smart_Pixels/Data_Files/Track_Lists/Signal", binsize=500, float_precision=5, overwrite=True)