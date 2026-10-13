# Incident Postmortem: Checkout Outage

## Summary
On February 12, 2024 the checkout service was unavailable for 47 minutes. Customers saw connection errors when trying to pay.

## Root Cause
The root cause was an expired TLS certificate on the public load balancer. Certificate renewal was a manual task and the reminder email went to a mailbox nobody monitored.

## Detection
Synthetic monitoring alerted the on-call engineer four minutes after the certificate expired.

## Action Items
- Automate certificate renewal with cert-manager.
- Alert 21 days before any certificate expires.
- Add the load balancer certificate to the weekly operations review.
