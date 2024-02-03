import glob
import subprocess

def set_usb_permissions():
    # Pattern to match all ttyUSBx devices
    usb_pattern = '/dev/ttyUSB*'
    
    # Find all paths matching the pattern
    usb_devices = glob.glob(usb_pattern)
    
    for device in usb_devices:
        try:
            # Use subprocess to call chmod and set permissions to 777
            subprocess.check_call(['sudo', 'chmod', '777', device])
            print(f"Permissions for {device} have been set to 777.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to set permissions for {device}. Error: {e}")

# Run the function
set_usb_permissions()
