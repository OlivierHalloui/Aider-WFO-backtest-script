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
journalctl -b --no-pager -n 50 | grep -E "(gdm|lightdm|nvidia|kernel|systemd|gpu|display)" || echo "No relevant boot logs found."
echo ""

# 3. Display Manager Status
echo "3. Display Manager Status:"
systemctl status gdm3 --no-pager || systemctl status lightdm --no-pager || echo "No display manager running or detected."
echo ""

# 4. NVIDIA Driver Info
echo "4. NVIDIA Driver and GPU Info:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available or no NVIDIA GPU detected."
lspci | grep -i nvidia || echo "No NVIDIA devices found via lspci."
echo ""

# 5. Xorg and Graphics
echo "5. Xorg and Graphics:"
ls /etc/X11/xorg.conf.d/ 2>/dev/null || echo "No Xorg config files found."
echo "Current DISPLAY: $DISPLAY"
echo "Current XAUTHORITY: $XAUTHORITY"
echo ""

# 6. Kernel Modules
echo "6. Loaded Kernel Modules (relevant to graphics):"
lsmod | grep -E "(nvidia|nouveau|radeon|amdgpu|i915)" || echo "No relevant graphics modules loaded."
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
echo ""

# 10. Recommendations
echo "10. Recommendations:"
if ! nvidia-smi >/dev/null 2>&1; then
    echo "- NVIDIA drivers may not be installed or configured. Try: sudo ubuntu-drivers autoinstall"
fi
if ! systemctl is-active --quiet gdm3; then
    echo "- gdm3 is not active. Check: sudo systemctl status gdm3"
fi
echo "- If screen freezes, try booting into recovery mode or switching to text mode (Ctrl+Alt+F1)."
echo "- For more help, check: https://ubuntu.com/desktop/drivers or NVIDIA forums."
echo ""

echo "=== End of Report ==="
