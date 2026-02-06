# Wi-Fi and Network Troubleshooting

## Office Wi-Fi Networks

### Available Networks
- **Corp-Secure**: Main corporate network (802.1X authentication)
- **Corp-Guest**: Guest network for visitors (web portal authentication)
- **Corp-IoT**: For IoT devices and printers (IT managed only)

### Connecting to Corp-Secure
1. Select "Corp-Secure" from available networks
2. Authentication: EAP-TLS or PEAP
3. Enter your company email and password
4. Trust the certificate when prompted
5. Connection should be automatic after first setup

## Common Network Issues

### Slow Internet
1. Run a speed test at https://speedtest.internal
2. Check if others in your area are affected
3. Try moving closer to a wireless access point
4. Connect via Ethernet if available (docking station)
5. If widespread, report to IT — may be an infrastructure issue

### Cannot Connect to Wi-Fi
- Forget the network and reconnect
- Restart Wi-Fi adapter (toggle off/on)
- Check if your device's MAC address is registered
- Try connecting to Corp-Guest to isolate the issue

### Printer Not Found on Network
1. Ensure printer is powered on and connected to Corp-IoT
2. Check printer IP from the label on the printer
3. Try adding printer by IP: Settings > Printers > Add > IP Address
4. Common printer server: `print.company.com`

## Remote Network Requirements
- VPN required for accessing internal resources
- Minimum bandwidth: 10 Mbps for video conferencing
- Recommended: Wired Ethernet connection for best performance
