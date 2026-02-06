# VPN Setup Guide

## Prerequisites
- Company laptop with admin access
- Active employee account
- VPN client software (Cisco AnyConnect or WireGuard)

## Setup Steps

### Windows
1. Download Cisco AnyConnect from the internal software portal at https://portal.internal/software
2. Install using default settings
3. Open AnyConnect and enter the VPN server address: `vpn.company.com`
4. Authenticate with your company email and password
5. Select the appropriate VPN profile (usually "Corporate Full Tunnel")

### macOS
1. Download Cisco AnyConnect from the internal software portal
2. Open the DMG and run the installer
3. Grant the required system permissions in System Settings > Privacy & Security
4. Launch AnyConnect from Applications
5. Enter server: `vpn.company.com`
6. Use your company credentials to connect

## Troubleshooting

### Connection Drops Frequently
- Check your internet connection stability
- Try switching between Wi-Fi and wired connection
- Restart the VPN client
- If on macOS, check that AnyConnect has network permissions

### "Authentication Failed" Error
- Verify your password hasn't expired
- Check if MFA is required and approve the push notification
- Contact IT if your account may be locked

### Split Tunnel vs Full Tunnel
- **Full Tunnel**: All traffic goes through VPN (more secure, slower)
- **Split Tunnel**: Only company traffic goes through VPN (faster for browsing)
- Request split tunnel access via IT ticket if needed
