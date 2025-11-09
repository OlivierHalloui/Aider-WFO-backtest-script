#!/bin/bash

# Ubuntu System Diagnostic Script
# This script checks various system components that might be related to boot issues, display problems, or NVIDIA driver conflicts.
# Run with: bash diagnose_ubuntu.sh

exec > >(tee report.txt) 2>&1

echo "=== Ubuntu System Diagnostic Report ==="
echo "Generated on: $(date)"
echo ""

# 1. OS and Kernel Info
echo "1. Operating System and Kernel:"
lsb_release -a 2>/dev/null || echo "lsb_release not available"
uname -a
echo ""

# 2. Boot Logs (last boot)
echo "2. Last Boot Logs (journalctl -b):"
journalctl -b --no-pager -n 50 | grep -E "(gdm|lightdm|nvidia|kernel|systemd|gpu|display|Xorg|wayland)" || echo "No relevant boot logs found."
echo ""

# 3. Display Manager Status
echo "3. Display Manager Status:"
systemctl status gdm3 --no-pager || systemctl status lightdm --no-pager || echo "No display manager running or detected."
echo "Checking for display manager logs..."
journalctl -u gdm3 --no-pager -n 20 | grep -E "(fail|error|session)" || journalctl -u lightdm --no-pager -n 20 | grep -E "(fail|error|session)" || echo "No display manager errors found in logs."
echo ""

# 4. NVIDIA Driver and GPU Info
echo "4. NVIDIA Driver and GPU Info:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available or no NVIDIA GPU detected."
lspci | grep -i nvidia || echo "No NVIDIA devices found via lspci."
echo "NVIDIA persistence mode:"
nvidia-smi -pm 1 2>/dev/null || echo "Unable to check persistence mode."
echo ""

# 5. Xorg and Graphics
echo "5. Xorg and Graphics:"
ls /etc/X11/xorg.conf.d/ 2>/dev/null || echo "No Xorg config files found."
echo "Current DISPLAY: $DISPLAY"
echo "Current XAUTHORITY: $XAUTHORITY"
echo "Checking Xorg log for errors..."
if [ -f /var/log/Xorg.0.log ]; then
    grep -i "error\|fail\|EE" /var/log/Xorg.0.log | tail -10 || echo "No errors in Xorg.0.log"
else
    echo "Xorg.0.log not found."
fi
echo "Session type (Wayland/X11):"
echo $XDG_SESSION_TYPE 2>/dev/null || echo "Session type not set (possibly text mode)."
echo ""

# 6. Kernel Modules
echo "6. Loaded Kernel Modules (relevant to graphics):"
lsmod | grep -E "(nvidia|nouveau|radeon|amdgpu|i915)" || echo "No relevant graphics modules loaded."
echo "Checking for module conflicts..."
lsmod | grep nouveau && echo "WARNING: Nouveau module loaded; may conflict with NVIDIA." || echo "No Nouveau conflicts detected."
echo ""

# 7. System Logs for Errors
echo "7. Recent System Errors (syslog):"
grep -i "error\|fail\|crash\|freeze" /var/log/syslog | tail -20 || echo "No recent errors in syslog."
echo ""

# 8. Disk Space and Memory
echo "8. Disk and Memory Usage:"
df -h /
free -h
echo ""

# 9. Package Status (NVIDIA related)
echo "9. NVIDIA Package Status:"
dpkg -l | grep -i nvidia || echo "No NVIDIA packages installed."
echo "Checking for residual packages..."
dpkg -l | grep -E "^rc" | grep -i nvidia || echo "No residual NVIDIA packages found."
echo ""

# 10. Secure Boot and Kernel
echo "10. Secure Boot and Kernel Integrity:"
mokutil --sb-state 2>/dev/null || echo "mokutil not available; secure boot status unknown."
echo "Kernel lockdown mode:"
cat /sys/kernel/security/lockdown 2>/dev/null || echo "Lockdown info not available."
echo ""

# 11. Recommendations
echo "11. Recommendations:"
if ! nvidia-smi >/dev/null 2>&1; then
    echo "- NVIDIA drivers may not be installed or configured. Try: sudo ubuntu-drivers autoinstall"
fi
if ! systemctl is-active --quiet gdm3 && ! systemctl is-active --quiet lightdm; then
    echo "- No display manager active. Try starting one: sudo systemctl start gdm3"
elif journalctl -u gdm3 --since "1 hour ago" | grep -q "fail"; then
    echo "- GDM errors detected. Consider switching to LightDM: sudo apt install lightdm && sudo dpkg-reconfigure lightdm"
fi
if [ -z "$DISPLAY" ]; then
    echo "- No graphical session detected. System may be in text mode; check recovery options."
fi
if lsmod | grep -q nouveau; then
    echo "- Nouveau driver loaded; blacklist it if using NVIDIA: add 'blacklist nouveau' to /etc/modprobe.d/blacklist.conf and update-initramfs."
fi
echo "- If screen freezes, try booting into recovery mode or switching to text mode (Ctrl+Alt+F1)."
echo "- For more help, check: https://ubuntu.com/desktop/drivers or NVIDIA forums."
echo ""

echo "=== End of Report ==="
