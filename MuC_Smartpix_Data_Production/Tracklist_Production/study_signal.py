from utils import getTracks
import subprocess

# Run set-up script
subprocess.run(['source', 'set_up.sh'], check=True)
print("Set up complete.")
 
# Run particle gun and detector simulation 
subprocess.run(['source', 'particle_gun.sh'], check=True)
print("Particle gun complete.")

subprocess.run(['source', 'detector_sim.sh'], check=True)
print("Detector simulation complete.")

plot = False

# Set up some options, constants
max_events = -1 # Set to -1 to run over all events
#Bfield = 3.57 # T for legacy

file_paths = "./output_sim.slcio"
getTracks(file_paths, allowedPIDS=[13], plot=plot, max_events=max_events, flp=0, tracklist_folder="Muon_Collider_Smart_Pixels/Data_Files/Track_Lists/Signal", binsize=500, float_precision=5, overwrite=True)