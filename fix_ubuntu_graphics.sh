#!/bin/bash

# Ubuntu Graphics Fix Script
# Fixes issues from diagnose_ubuntu.sh report: NVIDIA driver conflicts, display manager failures, and Xorg problems.
# Run as root in recovery/text mode. Reboot after completion.

set -e  # Exit on error
LOGFILE="/var/log/fix_graphics.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Ubuntu Graphics Fix Script ==="
echo "Started on: $(date)"
echo "Logging to: $LOGFILE"
echo ""

# Function to prompt user
prompt_user() {
    echo "$1"
    read -p "Continue? (y/N): " choice
    case "$choice" in
        y|Y ) echo "Proceeding...";;
        * ) echo "Aborting."; exit 1;;
    esac
}

# 1. Switch to text mode if graphical
if [ -n "$DISPLAY" ]; then
    echo "Graphical session detected. Switching to text mode..."
    sudo systemctl isolate multi-user.target
fi

# 2. Stop display manager
echo "Stopping display manager..."
sudo systemctl stop gdm3 || sudo systemctl stop lightdm || echo "No display manager to stop."

# 3. Purge old/conflicting NVIDIA packages
echo "Purging old NVIDIA packages and modules..."
sudo apt purge -y nvidia-* linux-modules-nvidia-* linux-objects-nvidia-* 2>/dev/null || echo "Some purges failed (expected if not installed)."
sudo apt autoremove -y
sudo apt autoclean

# 4. Update package lists
echo "Updating package lists..."
sudo apt update

# 5. Install recommended NVIDIA drivers
echo "Installing recommended NVIDIA drivers..."
sudo ubuntu-drivers autoinstall
# Alternatively, for specific version: sudo apt install nvidia-driver-580

# 6. Configure Xorg for NVIDIA (basic)
echo "Configuring Xorg..."
XORG_CONF="/etc/X11/xorg.conf"
if [ ! -f "$XORG_CONF" ]; then
    sudo nvidia-xconfig --no-logo --no-composite --no-expose-version
else
    echo "Xorg config exists; skipping generation. Manual check recommended."
fi

# 7. Blacklist Nouveau if not already (to prevent conflicts)
NOUVEAU_CONF="/etc/modprobe.d/nvidia-blacklist.conf"
if [ ! -f "$NOUVEAU_CONF" ]; then
    echo "blacklist nouveau" | sudo tee "$NOUVEAU_CONF"
    echo "options nouveau modeset=0" | sudo tee -a "$NOUVEAU_CONF"
    sudo update-initramfs -u
fi

# 8. Restart services
echo "Restarting services..."
sudo systemctl daemon-reload
sudo systemctl start gdm3 || sudo systemctl start lightdm || echo "Display manager start failed; check logs."

# 9. Final checks
echo "Final checks..."
nvidia-smi || echo "NVIDIA-SMI failed; drivers may need reboot."
lsmod | grep nvidia || echo "NVIDIA modules not loaded; reboot required."

echo ""
echo "=== Fix Complete ==="
echo "Reboot your system now: sudo reboot"
echo "After reboot, check: nvidia-smi, journalctl -b, and graphical login."
echo "If issues persist, try: sudo apt install nvidia-driver-XXX (replace XXX with version) or switch to Nouveau."
