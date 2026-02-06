# Email Troubleshooting

## Outlook Not Syncing

### Check Connection
1. Verify you have internet access
2. Check if VPN is required and connected
3. Try accessing Outlook Web at https://outlook.office365.com

### Reset Outlook Profile (Windows)
1. Close Outlook completely
2. Open Control Panel > Mail
3. Click "Show Profiles" > Remove the existing profile
4. Re-add your account using your company email
5. Outlook will re-download your mailbox

### Reset Outlook Profile (macOS)
1. Quit Outlook
2. Open Finder and navigate to ~/Library/Group Containers/UBF8T346G9.Office/Outlook/
3. Move the Outlook 15 Profiles folder to your Desktop as backup
4. Reopen Outlook and set up your account

## Cannot Send or Receive Emails

### Sending Issues
- Check if you're over your mailbox quota (50 GB limit)
- Verify the recipient address is correct
- Check your Outbox for stuck messages
- Large attachments (>25 MB) should use OneDrive sharing instead

### Receiving Issues
- Check Junk/Spam folder
- Verify mail flow rules haven't redirected your mail
- Ask the sender to check their bounce-back message

## Email Signature Setup
1. Open Outlook > Settings > Signatures
2. Use the company template from https://portal.internal/signature-generator
3. Include: Full name, title, department, phone, company logo

## Shared Mailbox Access
1. Request access via IT ticket
2. Manager approval required
3. Once granted, the shared mailbox appears automatically in Outlook
4. On mobile, add it as a separate account
