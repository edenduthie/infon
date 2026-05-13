"""
Enhanced NuExtract-2.0 template with full DIMEFILED taxonomy integration.

Incorporates:
  - DIMEFILED L1 (8 domains) + L2 (48 sub-domains) as constrained enums
  - L3 (112 codes) + L4 (829 codes) as constrained enums from DIMEFIL.md
  - 28 influence pattern tags (CSpade trajectory mining)
  - Structured actor model with country codes and organisation types
  - Structured location model with hierarchy levels and macro-regions
  - Edge relationships with strength and temporal precision
  - Multi-action extraction (one article -> multiple discrete actions)
  - Escalation classification and sovereignty context

Template + Schema pair:
  NUEXTRACT_TEMPLATE  -> semantic guidance (what to extract)
  XGRAMMAR_SCHEMA     -> structural constraint (valid JSON)
"""

# ============================================================================
# DIMEFILED L1 domains
# ============================================================================
L1_DOMAINS = [
    "D",    # DIPLOMATIC
    "I",    # INFORMATIONAL
    "M",    # MILITARY
    "E",    # ECONOMIC
    "F",    # FINANCIAL
    "IN",   # INTELLIGENCE
    "L",    # LAW ENFORCEMENT
    "EN",   # ENVIRONMENTAL
]

# ============================================================================
# DIMEFILED L2 sub-domains (48 codes)
# ============================================================================
L2_SUBDOMAINS = [
    # Diplomatic
    "D.1",   # Bilateral Relations
    "D.2",   # Multilateral Engagement
    "D.3",   # Diplomatic Pressure
    "D.4",   # Peace & Mediation
    "D.5",   # Normalization Operations
    # Informational
    "I.1",   # Strategic Comms
    "I.2",   # Information Operations
    "I.3",   # Cyber Information Operations
    "I.4",   # Media & Journalism
    # Military
    "M.1",   # Force Projection
    "M.2",   # Military Presence
    "M.3",   # Defense Cooperation
    "M.4",   # Unconventional Warfare
    "M.5",   # WMD & Strategic Systems
    "M.6",   # Space & Cyber Military
    "M.7",   # Escalation Management
    # Economic
    "E.1",   # Trade Policy
    "E.2",   # Economic Sanctions
    "E.3",   # Investment
    "E.4",   # Economic Coercion
    "E.5",   # Economic Cooperation
    "E.6",   # Resource & Energy
    "E.7",   # Economic Restructuring
    # Financial
    "F.1",   # Financial Policy
    "F.2",   # Banking & Payment Systems
    "F.3",   # Capital Markets
    "F.4",   # International Finance
    "F.5",   # Financial Crime & Enforcement
    "F.6",   # Financial Innovation & Disruption
    # Intelligence
    "IN.1",  # HUMINT
    "IN.2",  # SIGINT
    "IN.3",  # Cyber Intelligence
    "IN.4",  # Technical Intelligence
    "IN.5",  # Counterintelligence
    "IN.6",  # Covert Action
    "IN.7",  # Intelligence Cooperation
    "IN.8",  # Attribution Warfare
    # Law Enforcement
    "L.1",   # Transnational Crime
    "L.2",   # International Law Enforcement Cooperation
    "L.3",   # Border & Immigration
    "L.4",   # Maritime Law
    "L.5",   # Legal Warfare (Lawfare)
    "L.6",   # Regulatory Enforcement
    "L.7",   # Judicial Cooperation
    "L.8",   # Transnational Repression
    # Environmental
    "EN.1",  # Water Warfare
    "EN.2",  # Atmospheric & Climate Operations
    "EN.3",  # Biological Resource Warfare
]

# ============================================================================
# DIMEFILED L3 action categories (from DIMEFIL.md)
# ============================================================================
L3_SUBDOMAINS = [
    # D
    "State Visits & Summits",
    "Diplomatic Recognition",
    "Embassy & Consular Operations",
    "Bilateral Agreements",
    "International Organizations",
    "Regional Organizations",
    "Multilateral Treaties",
    "Procedural Warfare",
    "Formal Protests",
    "Diplomatic Isolation",
    "Conflict Resolution",
    "Mediation & Facilitation",
    "Threshold Shifting",
    "Legitimization Campaigns",

    # I
    "Public Diplomacy",
    "Government Messaging",
    "Propaganda & Influence",
    "Disinformation & Deception",
    "Information Warfare",
    "Cyber-Enabled Information Ops",
    "Information Control",
    "Leak Ecosystem Operations",
    "Media Relations",
    "Media Control",

    # M
    "Combat Operations",
    "Military Posturing",
    "Naval Operations",
    "Air Operations",
    "Maritime Gray Zone Harassment",
    "Forward Deployment",
    "Base Operations",
    "Military Assistance",
    "Alliance Operations",
    "Proxy Operations",
    "Irregular Warfare",
    "Nuclear Operations",
    "Missile Operations",
    "CBRN Operations",
    "Space Operations",
    "Cyber Military Operations",
    "Escalation Control",
    "De-escalation Operations",

    # E
    "Tariffs & Duties",
    "Trade Agreements",
    "Import/Export Controls",
    "Comprehensive Sanctions",
    "Targeted Sanctions",
    "Export Controls",
    "Humanitarian Economic Warfare",
    "Foreign Direct Investment",
    "Development Finance",
    "Sovereign Debt Weaponization",
    "Market Access",
    "Supply Chain",
    "Chokepoint Control",
    "Economic Integration",
    "Industrial Cooperation",
    "Energy Policy",
    "Resource Control",
    "Decoupling Operations",
    "Re-coupling Operations",

    # F
    "Currency Operations",
    "Central Bank Actions",
    "Banking Restrictions",
    "Alternative Payment Systems",
    "Financial Market Access",
    "Market Manipulation",
    "Multilateral Finance",
    "Debt Operations",
    "Anti-Money Laundering",
    "Asset Recovery",
    "Fintech Weaponization",
    "Financial Infrastructure",

    # IN
    "Agent Operations",
    "Diplomatic Intelligence",
    "Communications Intelligence",
    "Electronic Intelligence",
    "Cyber Espionage",
    "Cyber Surveillance",
    "Imagery Intelligence (IMINT)",
    "Measurement & Signature Intelligence",
    "Defensive CI",
    "Offensive CI",
    "Political Action",
    "Paramilitary Action",
    "Intelligence Sharing",
    "Joint Operations",
    "Attribution Obfuscation",
    "Attribution Shaping",

    # L
    "Organized Crime",
    "Terrorism",
    "Law Enforcement Bilateral Cooperation",
    "Law Enforcement Multilateral Cooperation",
    "Border Control",
    "Immigration Enforcement",
    "Maritime Security",
    "Coast Guard Operations",
    "Jurisdictional Assertion",
    "Legal Persecution",
    "Temporal Legal Operations",
    "Democratic Process Weaponization",
    "Economic Regulation",
    "Technology Regulation",
    "International Courts",
    "Legal Assistance",
    "Extraterritorial Law Enforcement",
    "Diaspora Control",

    # EN
    "Hydro-Hegemony Operations",
    "Weather Modification",
    "Pollution Export",
    "Fisheries Depletion",
    "Agricultural Warfare",
]

# Name -> code lookup table for postprocessing
L3_NAME_TO_CODE = {
    "State Visits & Summits": "D.1.1",
    "Diplomatic Recognition": "D.1.2",
    "Embassy & Consular Operations": "D.1.3",
    "Bilateral Agreements": "D.1.4",
    "International Organizations": "D.2.1",
    "Regional Organizations": "D.2.2",
    "Multilateral Treaties": "D.2.3",
    "Procedural Warfare": "D.2.4",
    "Formal Protests": "D.3.1",
    "Diplomatic Isolation": "D.3.2",
    "Conflict Resolution": "D.4.1",
    "Mediation & Facilitation": "D.4.2",
    "Threshold Shifting": "D.5.1",
    "Legitimization Campaigns": "D.5.2",
    "Public Diplomacy": "I.1.1",
    "Government Messaging": "I.1.2",
    "Propaganda & Influence": "I.2.1",
    "Disinformation & Deception": "I.2.2",
    "Information Warfare": "I.2.3",
    "Cyber-Enabled Information Ops": "I.3.1",
    "Information Control": "I.3.2",
    "Leak Ecosystem Operations": "I.3.3",
    "Media Relations": "I.4.1",
    "Media Control": "I.4.2",
    "Combat Operations": "M.1.1",
    "Military Posturing": "M.1.2",
    "Naval Operations": "M.1.3",
    "Air Operations": "M.1.4",
    "Maritime Gray Zone Harassment": "M.1.5",
    "Forward Deployment": "M.2.1",
    "Base Operations": "M.2.2",
    "Military Assistance": "M.3.1",
    "Alliance Operations": "M.3.2",
    "Proxy Operations": "M.4.1",
    "Irregular Warfare": "M.4.2",
    "Nuclear Operations": "M.5.1",
    "Missile Operations": "M.5.2",
    "CBRN Operations": "M.5.3",
    "Space Operations": "M.6.1",
    "Cyber Military Operations": "M.6.2",
    "Escalation Control": "M.7.1",
    "De-escalation Operations": "M.7.2",
    "Tariffs & Duties": "E.1.1",
    "Trade Agreements": "E.1.2",
    "Import/Export Controls": "E.1.3",
    "Comprehensive Sanctions": "E.2.1",
    "Targeted Sanctions": "E.2.2",
    "Export Controls": "E.2.3",
    "Humanitarian Economic Warfare": "E.2.4",
    "Foreign Direct Investment": "E.3.1",
    "Development Finance": "E.3.2",
    "Sovereign Debt Weaponization": "E.3.3",
    "Market Access": "E.4.1",
    "Supply Chain": "E.4.2",
    "Chokepoint Control": "E.4.3",
    "Economic Integration": "E.5.1",
    "Industrial Cooperation": "E.5.2",
    "Energy Policy": "E.6.1",
    "Resource Control": "E.6.2",
    "Decoupling Operations": "E.7.1",
    "Re-coupling Operations": "E.7.2",
    "Currency Operations": "F.1.1",
    "Central Bank Actions": "F.1.2",
    "Banking Restrictions": "F.2.1",
    "Alternative Payment Systems": "F.2.2",
    "Financial Market Access": "F.3.1",
    "Market Manipulation": "F.3.2",
    "Multilateral Finance": "F.4.1",
    "Debt Operations": "F.4.2",
    "Anti-Money Laundering": "F.5.1",
    "Asset Recovery": "F.5.2",
    "Fintech Weaponization": "F.6.1",
    "Financial Infrastructure": "F.6.2",
    "Agent Operations": "IN.1.1",
    "Diplomatic Intelligence": "IN.1.2",
    "Communications Intelligence": "IN.2.1",
    "Electronic Intelligence": "IN.2.2",
    "Cyber Espionage": "IN.3.1",
    "Cyber Surveillance": "IN.3.2",
    "Imagery Intelligence (IMINT)": "IN.4.1",
    "Measurement & Signature Intelligence": "IN.4.2",
    "Defensive CI": "IN.5.1",
    "Offensive CI": "IN.5.2",
    "Political Action": "IN.6.1",
    "Paramilitary Action": "IN.6.2",
    "Intelligence Sharing": "IN.7.1",
    "Joint Operations": "IN.7.2",
    "Attribution Obfuscation": "IN.8.1",
    "Attribution Shaping": "IN.8.2",
    "Organized Crime": "L.1.1",
    "Terrorism": "L.1.2",
    "Law Enforcement Bilateral Cooperation": "L.2.1",
    "Law Enforcement Multilateral Cooperation": "L.2.2",
    "Border Control": "L.3.1",
    "Immigration Enforcement": "L.3.2",
    "Maritime Security": "L.4.1",
    "Coast Guard Operations": "L.4.2",
    "Jurisdictional Assertion": "L.5.1",
    "Legal Persecution": "L.5.2",
    "Temporal Legal Operations": "L.5.3",
    "Democratic Process Weaponization": "L.5.4",
    "Economic Regulation": "L.6.1",
    "Technology Regulation": "L.6.2",
    "International Courts": "L.7.1",
    "Legal Assistance": "L.7.2",
    "Extraterritorial Law Enforcement": "L.8.1",
    "Diaspora Control": "L.8.2",
    "Hydro-Hegemony Operations": "EN.1.1",
    "Weather Modification": "EN.2.1",
    "Pollution Export": "EN.2.2",
    "Fisheries Depletion": "EN.3.1",
    "Agricultural Warfare": "EN.3.2",
}

# ============================================================================
# DIMEFILED L4 action subtypes (from DIMEFIL.md)
# ============================================================================
L4_SUBDOMAINS = [
    # D.1
    "D.1.1.1 - Head of State Official Visit",
    "D.1.1.2 - Head of Government Working Visit",
    "D.1.1.3 - Bilateral Summit Meeting",
    "D.1.1.4 - Emergency Leadership Consultation",
    "D.1.1.5 - Virtual/Remote Bilateral Summit",
    "D.1.1.6 - Unofficial/Back-channel Leadership Meeting",
    "D.1.1.7 - United Front Delegation Visit",
    "D.1.1.8 - Party-to-Party Diplomatic Engagement",
    "D.1.2.1 - Establishment of Diplomatic Relations",
    "D.1.2.2 - Severance of Diplomatic Relations",
    "D.1.2.3 - Downgrading of Diplomatic Relations",
    "D.1.2.4 - Restoration of Diplomatic Relations",
    "D.1.2.5 - Recognition of Government/Regime",
    "D.1.2.6 - Withdrawal of Recognition",
    "D.1.2.7 - Recognition of Territorial Claims",
    "D.1.2.8 - Non-recognition Declaration",
    "D.1.2.9 - Diplomatic Recognition Hostage-Taking",
    "D.1.3.1 - Embassy Opening/Establishment",
    "D.1.3.2 - Embassy Closure",
    "D.1.3.3 - Embassy Staff Expulsion",
    "D.1.3.4 - Embassy Staff Withdrawal",
    "D.1.3.5 - Consulate Opening",
    "D.1.3.6 - Consulate Closure",
    "D.1.3.7 - Ambassador Appointment",
    "D.1.3.8 - Ambassador Recall",
    "D.1.3.9 - Persona Non Grata Declaration",
    "D.1.3.10 - Diplomatic Asylum Granted",
    "D.1.3.11 - Diplomatic Immunity Waiver",
    "D.1.3.12 - Diplomatic Facility Surveillance Harassment",
    "D.1.4.1 - Treaty Signing",
    "D.1.4.2 - Treaty Ratification",
    "D.1.4.3 - Treaty Withdrawal/Abrogation",
    "D.1.4.4 - Memorandum of Understanding Signing",
    "D.1.4.5 - Executive Agreement",
    "D.1.4.6 - Status of Forces Agreement",
    "D.1.4.7 - Bilateral Trade Agreement",
    "D.1.4.8 - Mutual Defense Treaty",
    "D.1.4.9 - Non-Aggression Pact",
    "D.1.4.10 - Extradition Treaty",
    "D.1.4.11 - Subnational Government Agreement",
    "D.1.4.12 - Sister City Exploitation for Intelligence",

    # D.2
    "D.2.1.1 - UN Security Council Resolution Proposal",
    "D.2.1.2 - UN General Assembly Address",
    "D.2.1.3 - UN Voting/Abstention",
    "D.2.1.4 - UN Peacekeeping Contribution",
    "D.2.1.5 - International Organization Membership Application",
    "D.2.1.6 - International Organization Withdrawal",
    "D.2.1.7 - International Organization Leadership Bid",
    "D.2.1.8 - Blocking International Organization Action",
    "D.2.1.9 - International Court Filing",
    "D.2.1.10 - International Court Judgment Compliance/Non-compliance",
    "D.2.1.11 - Alternative Institution Creation to Bypass Existing Order",
    "D.2.1.12 - Institutional Rule Interpretation Manipulation",
    "D.2.1.13 - International Standard Setting Capture",
    "D.2.1.14 - UN Agency Leadership Campaign",
    "D.2.2.1 - Regional Organization Summit Participation",
    "D.2.2.2 - Regional Organization Membership Application",
    "D.2.2.3 - Regional Organization Withdrawal",
    "D.2.2.4 - Regional Organization Initiative Proposal",
    "D.2.2.5 - Regional Organization Veto/Blocking Action",
    "D.2.2.6 - Regional Organization Sanctions Support",
    "D.2.2.7 - Regional Integration Agreement",
    "D.2.3.1 - Multilateral Treaty Signature",
    "D.2.3.2 - Multilateral Treaty Ratification",
    "D.2.3.3 - Multilateral Treaty Withdrawal",
    "D.2.3.4 - Treaty Reservation Declaration",
    "D.2.3.5 - Treaty Protocol Addition",
    "D.2.3.6 - Arms Control Agreement Participation",
    "D.2.3.7 - Environmental Agreement Participation",
    "D.2.3.8 - Non-Proliferation Agreement Signature",
    "D.2.4.1 - No-Action Motion Deployment",
    "D.2.4.2 - Committee Agenda Control",
    "D.2.4.3 - GONGO Swarming",
    "D.2.4.4 - NGO Accreditation Blocking",
    "D.2.4.5 - Expert Panel Capture",
    "D.2.4.6 - Resolution Language Dilution",
    "D.2.4.7 - Technical Hold Abuse",

    # D.3
    "D.3.1.1 - Diplomatic Note of Protest",
    "D.3.1.2 - Démarche Delivery",
    "D.3.1.3 - Official Condemnation Statement",
    "D.3.1.4 - Recall of Ambassador for Consultations",
    "D.3.1.5 - Summoning of Foreign Ambassador",
    "D.3.1.6 - Joint International Condemnation",
    "D.3.1.7 - Diplomatic Boycott",
    "D.3.2.1 - International Coalition Building Against Target",
    "D.3.2.2 - Diplomatic Quarantine",
    "D.3.2.3 - International Forum Exclusion",
    "D.3.2.4 - Withdrawal of Diplomatic Privileges",
    "D.3.2.5 - Travel Ban on Officials",
    "D.3.2.6 - Cancellation of State Visits",
    "D.3.2.7 - Diplomatic Passport/Visa Weaponization",
    "D.3.2.8 - Academic/Cultural Exchange Restriction",

    # D.4
    "D.4.1.1 - Peace Process Initiation",
    "D.4.1.2 - Ceasefire Negotiation",
    "D.4.1.3 - Peace Treaty Signing",
    "D.4.1.4 - Armistice Agreement",
    "D.4.1.5 - Humanitarian Corridor Negotiation",
    "D.4.1.6 - Prisoner Exchange Negotiation",
    "D.4.1.7 - Hostage Release Negotiation",
    "D.4.2.1 - Third-Party Mediation Offer",
    "D.4.2.2 - Good Offices Provision",
    "D.4.2.3 - Shuttle Diplomacy",
    "D.4.2.4 - Track II Diplomacy Initiative",
    "D.4.2.5 - Arbitration Agreement",
    "D.4.2.6 - Fact-Finding Mission",
    "D.4.2.7 - Peace Conference Hosting",
    "D.4.2.8 - Biased Mediation Offer",
    "D.4.2.9 - Frozen Conflict Creation",

    # D.5
    "D.5.1.1 - Gradual Threshold Erosion",
    "D.5.1.2 - Precedent Creation",
    "D.5.1.3 - Norm Entrepreneurship",
    "D.5.1.4 - Overton Window Manipulation",
    "D.5.2.1 - Historical Justification Campaign",
    "D.5.2.2 - Legal Reinterpretation Drive",
    "D.5.2.3 - Academic Legitimization",
    "D.5.2.4 - International Acceptance Campaign",

    # I.1
    "I.1.1.1 - International Broadcasting",
    "I.1.1.2 - Cultural Center Establishment",
    "I.1.1.3 - Educational Exchange Program",
    "I.1.1.4 - Language Program Promotion",
    "I.1.1.5 - International Media Interview Campaign",
    "I.1.1.6 - Public Opinion Polling/Research",
    "I.1.1.7 - Nation Branding Campaign",
    "I.1.1.8 - Diaspora Engagement Program",
    "I.1.1.9 - Diaspora Weaponization",
    "I.1.1.10 - Academic Institution Infiltration",
    "I.1.1.11 - False Persona Creation",
    "I.1.2.1 - Official Government Statement",
    "I.1.2.2 - Press Conference/Briefing",
    "I.1.2.3 - White Paper/Policy Document Release",
    "I.1.2.4 - Emergency Broadcasting",
    "I.1.2.5 - National Address",
    "I.1.2.6 - Parliamentary/Congressional Statement",
    "I.1.2.7 - Ministry/Department Announcement",

    # I.2
    "I.2.1.1 - State Media Narrative Campaign",
    "I.2.1.2 - Social Media Influence Operation",
    "I.2.1.3 - Bot Network Deployment",
    "I.2.1.4 - Astroturfing Campaign",
    "I.2.1.5 - False Grassroots Movement Creation",
    "I.2.1.6 - Paid Influencer Campaign",
    "I.2.1.7 - Front Organization Establishment",
    "I.2.1.8 - Covert Media Outlet Operation",
    "I.2.1.9 - Artificial Amplification of Authentic Voices",
    "I.2.1.10 - Microtargeted Polarization Campaigns",
    "I.2.2.1 - Disinformation Campaign Launch",
    "I.2.2.2 - Deepfake Content Deployment",
    "I.2.2.3 - False Flag Information Operation",
    "I.2.2.4 - Document Forgery Release",
    "I.2.2.5 - Conspiracy Theory Promotion",
    "I.2.2.6 - Historical Revisionism Campaign",
    "I.2.2.7 - False Attribution Operation",
    "I.2.2.8 - Synthetic Media Creation",
    "I.2.2.9 - Selective Document Manipulation",
    "I.2.3.1 - Cognitive Warfare Operation",
    "I.2.3.2 - Psychological Operation (PSYOP)",
    "I.2.3.3 - Narrative Warfare Campaign",
    "I.2.3.4 - Memetic Warfare",
    "I.2.3.5 - Information Blockade",
    "I.2.3.6 - Counter-Narrative Campaign",
    "I.2.3.7 - Reflexive Control Operation",
    "I.2.3.8 - Computational Propaganda",
    "I.2.3.9 - Legal Warfare Information Campaign",

    # I.3
    "I.3.1.1 - Website Defacement",
    "I.3.1.2 - Social Media Account Hijacking",
    "I.3.1.3 - Email Leak/Hack and Release",
    "I.3.1.4 - Database Breach and Disclosure",
    "I.3.1.5 - Cyber-Enabled Doxxing",
    "I.3.1.6 - Search Engine Manipulation",
    "I.3.1.7 - Platform Algorithm Gaming",
    "I.3.1.8 - IoT Device Information Weaponization",
    "I.3.2.1 - Internet Shutdown",
    "I.3.2.2 - Social Media Platform Ban",
    "I.3.2.3 - Website Blocking/Filtering",
    "I.3.2.4 - DNS Hijacking",
    "I.3.2.5 - Content Moderation Manipulation",
    "I.3.2.6 - Bandwidth Throttling",
    "I.3.2.7 - VPN Blocking",
    "I.3.2.8 - Shadow Banning/Algorithmic Suppression",
    "I.3.3.1 - Coordinated Media Leak Operation",
    "I.3.3.2 - Journalist Asset Cultivation",
    "I.3.3.3 - Leak Platform Creation/Control",
    "I.3.3.4 - Selective Document Curation",
    "I.3.3.5 - Timed Release Coordination",
    "I.3.3.6 - Parallel Leak Campaign",

    # I.4
    "I.4.1.1 - Exclusive Interview/Access Provision",
    "I.4.1.2 - Press Embargo",
    "I.4.1.3 - Media Accreditation Control",
    "I.4.1.4 - Press Pool Manipulation",
    "I.4.1.5 - Journalist Visa Denial/Revocation",
    "I.4.1.6 - Foreign Correspondent Expulsion",
    "I.4.2.1 - Media Outlet Closure",
    "I.4.2.2 - Broadcast License Revocation",
    "I.4.2.3 - Censorship Directive",
    "I.4.2.4 - Editorial Control Imposition",
    "I.4.2.5 - Journalist Arrest/Detention",
    "I.4.2.6 - Media Ownership Forced Transfer",
    "I.4.2.7 - Advertising Boycott Organization",
    "I.4.2.8 - Content Farm Creation",
    "I.4.2.9 - Local Media Acquisition Through Proxies",

    # M.1
    "M.1.1.1 - Full-Scale Military Invasion",
    "M.1.1.2 - Limited Military Strike",
    "M.1.1.3 - Surgical/Precision Strike",
    "M.1.1.4 - Artillery Bombardment",
    "M.1.1.5 - Naval Bombardment",
    "M.1.1.6 - Aerial Bombing Campaign",
    "M.1.1.7 - Missile Strike",
    "M.1.1.8 - Drone Strike Operation",
    "M.1.1.9 - Cross-Border Raid",
    "M.1.1.10 - Special Operations Raid",
    "M.1.1.11 - Amphibious Assault",
    "M.1.1.12 - Airborne Operation",
    "M.1.1.13 - Calibrated Escalation Strike",
    "M.1.1.14 - Deniable Kinetic Operation",
    "M.1.1.15 - Autonomous Swarm Deployment",
    "M.1.1.16 - Loitering Munition Strike",
    "M.1.2.1 - Military Exercise/Drill",
    "M.1.2.2 - Joint Military Exercise",
    "M.1.2.3 - Snap/Unscheduled Exercise",
    "M.1.2.4 - Nuclear Forces Exercise",
    "M.1.2.5 - Live-Fire Exercise",
    "M.1.2.6 - Military Parade/Demonstration",
    "M.1.2.7 - Weapons Test/Demonstration",
    "M.1.2.8 - Force Mobilization",
    "M.1.2.9 - Reserve Activation",
    "M.1.2.10 - Military Alert Level Change",
    "M.1.3.1 - Freedom of Navigation Operation",
    "M.1.3.2 - Naval Blockade",
    "M.1.3.3 - Maritime Interdiction",
    "M.1.3.4 - Carrier Strike Group Deployment",
    "M.1.3.5 - Submarine Patrol",
    "M.1.3.6 - Naval Escort Operation",
    "M.1.3.7 - Mine Laying Operation",
    "M.1.3.8 - Mine Clearing Operation",
    "M.1.3.9 - Anti-Piracy Operation",
    "M.1.3.10 - Naval Port Visit",
    "M.1.4.1 - No-Fly Zone Establishment",
    "M.1.4.2 - Air Patrol/CAP Mission",
    "M.1.4.3 - Aerial Reconnaissance",
    "M.1.4.4 - Air Interdiction",
    "M.1.4.5 - Close Air Support",
    "M.1.4.6 - Strategic Bomber Deployment",
    "M.1.4.7 - Airlift Operation",
    "M.1.4.8 - Air-to-Air Engagement",
    "M.1.4.9 - Suppression of Air Defenses",
    "M.1.4.10 - Electronic Warfare Flight",
    "M.1.5.1 - Physical Ship Harassment (Shouldering/Ramming)",
    "M.1.5.2 - Wake/Wash Attacks",
    "M.1.5.3 - Laser Dazzler Employment",
    "M.1.5.4 - LRAD Sonic Attacks",
    "M.1.5.5 - Cable/Array Cutting",
    "M.1.5.6 - Net Fouling Operations",
    "M.1.5.7 - Fishing Fleet Swarming",
    "M.1.5.8 - Cabbage Strategy Blockade",
    "M.1.5.9 - Maritime Militia Rafting",
    "M.1.5.10 - Debris Field Creation",

    # M.2
    "M.2.1.1 - Permanent Base Establishment",
    "M.2.1.2 - Forward Operating Base Creation",
    "M.2.1.3 - Lily Pad/Cooperative Security Location",
    "M.2.1.4 - Troop Rotation/Deployment",
    "M.2.1.5 - Pre-positioned Equipment Placement",
    "M.2.1.6 - Military Advisor Deployment",
    "M.2.1.7 - Training Mission Establishment",
    "M.2.1.8 - Dual-Use Infrastructure Development",
    "M.2.1.9 - Strategic Strongpoint Creation",
    "M.2.2.1 - Base Expansion/Upgrade",
    "M.2.2.2 - Base Closure",
    "M.2.2.3 - Base Access Agreement",
    "M.2.2.4 - Port Access Agreement",
    "M.2.2.5 - Overflight Rights Agreement",
    "M.2.2.6 - Logistics Hub Establishment",
    "M.2.2.7 - Joint Base Operation",
    "M.2.2.8 - Snap Base Activation",
    "M.2.2.9 - Base Access Ambiguity",

    # M.3
    "M.3.1.1 - Foreign Military Sales",
    "M.3.1.2 - Foreign Military Financing",
    "M.3.1.3 - Excess Defense Articles Transfer",
    "M.3.1.4 - Military Grant Aid",
    "M.3.1.5 - Training & Equipping Program",
    "M.3.1.6 - Defense Capacity Building",
    "M.3.1.7 - Military Education Exchange",
    "M.3.1.8 - Security Force Assistance Below Armed Conflict",
    "M.3.1.9 - Volunteer Fighter Facilitation",
    "M.3.2.1 - Alliance Formation",
    "M.3.2.2 - Alliance Expansion",
    "M.3.2.3 - Alliance Article 5/Collective Defense Invocation",
    "M.3.2.4 - Allied Force Integration",
    "M.3.2.5 - Burden Sharing Negotiation",
    "M.3.2.6 - Alliance Withdrawal Threat",
    "M.3.2.7 - Combined Joint Task Force Creation",

    # M.4
    "M.4.1.1 - Proxy Force Arming",
    "M.4.1.2 - Proxy Force Training",
    "M.4.1.3 - Proxy Force Financing",
    "M.4.1.4 - Militia Creation/Support",
    "M.4.1.5 - Foreign Fighter Recruitment",
    "M.4.1.6 - Mercenary Deployment",
    "M.4.1.7 - Private Military Contractor Use",
    "M.4.1.8 - Cyber Militia Activation",
    "M.4.2.1 - Guerrilla Warfare Support",
    "M.4.2.2 - Insurgency Support",
    "M.4.2.3 - Counter-Insurgency Operation",
    "M.4.2.4 - Sabotage Operation",
    "M.4.2.5 - Subversion Campaign",
    "M.4.2.6 - Resistance Movement Support",
    "M.4.2.7 - Coup Support/Attempt",
    "M.4.2.8 - Little Green Men Deployment",
    "M.4.2.9 - Plausibly Deniable Sabotage",

    # M.5
    "M.5.1.1 - Nuclear Weapons Test",
    "M.5.1.2 - Nuclear Alert Status Change",
    "M.5.1.3 - Nuclear Deployment/Forward Positioning",
    "M.5.1.4 - Nuclear Doctrine Announcement",
    "M.5.1.5 - Nuclear Umbrella Extension",
    "M.5.1.6 - Nuclear Sharing Agreement",
    "M.5.1.7 - Nuclear Threshold Signaling",
    "M.5.1.8 - Nuclear Ambiguity Signaling",
    "M.5.1.9 - Dual-Capable System Deployment",
    "M.5.2.1 - Ballistic Missile Test",
    "M.5.2.2 - Cruise Missile Test",
    "M.5.2.3 - Hypersonic Weapon Test",
    "M.5.2.4 - Anti-Satellite Weapon Test",
    "M.5.2.5 - Missile Defense Deployment",
    "M.5.2.6 - Missile System Sale/Transfer",
    "M.5.3.1 - Chemical Weapon Use",
    "M.5.3.2 - Biological Weapon Deployment",
    "M.5.3.3 - Radiological Weapon Use",
    "M.5.3.4 - CBRN Defense Exercise",
    "M.5.3.5 - CBRN Threat/Blackmail",

    # M.6
    "M.6.1.1 - Military Satellite Launch",
    "M.6.1.2 - Anti-Satellite Operation",
    "M.6.1.3 - Space Weapon Deployment",
    "M.6.1.4 - Satellite Jamming/Spoofing",
    "M.6.1.5 - Space Surveillance Operation",
    "M.6.1.6 - Orbital Rendezvous/Proximity Operation",
    "M.6.1.7 - Rendezvous and Proximity Operations Below Attack Threshold",
    "M.6.1.8 - Reversible Space Interference",
    "M.6.2.1 - Offensive Cyber Operation",
    "M.6.2.2 - Cyber Espionage Operation",
    "M.6.2.3 - Critical Infrastructure Cyber Attack",
    "M.6.2.4 - Military Network Infiltration",
    "M.6.2.5 - Cyber Command Establishment",
    "M.6.2.6 - Cyber Defense Exercise",
    "M.6.2.7 - Pre-Positioned Cyber Implant",
    "M.6.2.8 - Supply Chain Hardware Compromise",

    # M.7
    "M.7.1.1 - Controlled Escalation Signal",
    "M.7.1.2 - Escalation Pause/Freeze",
    "M.7.1.3 - Proportional Response",
    "M.7.1.4 - Escalation Dominance Display",
    "M.7.2.1 - Face-Saving Off-Ramp",
    "M.7.2.2 - Mutual Step-Back Agreement",
    "M.7.2.3 - Third-Party De-escalation",
    "M.7.2.4 - Confidence Building Measure",

    # E.1
    "E.1.1.1 - Tariff Imposition/Increase",
    "E.1.1.2 - Tariff Reduction/Elimination",
    "E.1.1.3 - Punitive/Retaliatory Tariff",
    "E.1.1.4 - Anti-Dumping Duty",
    "E.1.1.5 - Countervailing Duty",
    "E.1.1.6 - Safeguard Measure",
    "E.1.1.7 - Tariff Rate Quota Implementation",
    "E.1.1.8 - Preferential Tariff Grant",
    "E.1.2.1 - Free Trade Agreement Negotiation",
    "E.1.2.2 - Customs Union Formation",
    "E.1.2.3 - Common Market Establishment",
    "E.1.2.4 - Trade Agreement Withdrawal",
    "E.1.2.5 - Trade Agreement Renegotiation",
    "E.1.2.6 - Bilateral Investment Treaty",
    "E.1.2.7 - Double Taxation Agreement",
    "E.1.2.8 - Trade Facilitation Agreement",
    "E.1.3.1 - Export Ban/Restriction",
    "E.1.3.2 - Import Ban/Restriction",
    "E.1.3.3 - Quota Implementation",
    "E.1.3.4 - Licensing Requirement Imposition",
    "E.1.3.5 - Export Subsidy",
    "E.1.3.6 - Import Substitution Policy",
    "E.1.3.7 - Local Content Requirement",
    "E.1.3.8 - Rules of Origin Change",
    "E.1.3.9 - Informal Trade Restriction",
    "E.1.3.10 - Quality/Safety Standard Weaponization",
    "E.1.3.11 - Forced Technology Transfer Requirement",
    "E.1.3.12 - Indigenous Innovation Preference",

    # E.2
    "E.2.1.1 - Full Economic Embargo",
    "E.2.1.2 - Trade Embargo",
    "E.2.1.3 - Financial Embargo",
    "E.2.1.4 - Arms Embargo",
    "E.2.1.5 - Travel Ban",
    "E.2.1.6 - Comprehensive UN Sanctions",
    "E.2.1.7 - Regional Organization Sanctions",
    "E.2.2.1 - Sectoral Sanctions",
    "E.2.2.2 - Smart/Targeted Individual Sanctions",
    "E.2.2.3 - Entity List Addition",
    "E.2.2.4 - Asset Freeze",
    "E.2.2.5 - Transaction Ban",
    "E.2.2.6 - Technology Transfer Ban",
    "E.2.2.7 - Services Prohibition",
    "E.2.2.8 - Secondary Sanctions",
    "E.2.2.9 - Sanctions Threat/Signaling",
    "E.2.2.10 - Voluntary Compliance Pressure",
    "E.2.3.1 - Dual-Use Technology Control",
    "E.2.3.2 - Military Equipment Export Ban",
    "E.2.3.3 - High-Tech Export Restriction",
    "E.2.3.4 - Critical Materials Export Control",
    "E.2.3.5 - Software/IP Export Control",
    "E.2.3.6 - Re-export Restriction",
    "E.2.3.7 - End-User Verification",
    "E.2.3.8 - Cloud Computing Service Denial",
    "E.2.4.1 - Humanitarian Exemption Manipulation",
    "E.2.4.2 - Medical Supply Conditionality",
    "E.2.4.3 - Food Security Weaponization",
    "E.2.4.4 - Humanitarian Aid Diversion",
    "E.2.4.5 - NGO Financial Restriction",
    "E.2.4.6 - Humanitarian Corridor Blockade",

    # E.3
    "E.3.1.1 - FDI Restriction/Screening",
    "E.3.1.2 - FDI Promotion/Incentive",
    "E.3.1.3 - Sovereign Wealth Fund Investment",
    "E.3.1.4 - State-Owned Enterprise Acquisition",
    "E.3.1.5 - Forced Divestment Order",
    "E.3.1.6 - Golden Power/Veto Exercise",
    "E.3.1.7 - Investment Treaty Claim",
    "E.3.1.8 - Nationalization/Expropriation",
    "E.3.1.9 - Strategic Sector Investment Screening Expansion",
    "E.3.1.10 - Hidden State Ownership Detection",
    "E.3.2.1 - Development Loan/Grant",
    "E.3.2.2 - Infrastructure Investment",
    "E.3.2.3 - Belt and Road/Connectivity Initiative",
    "E.3.2.4 - Debt Trap Creation",
    "E.3.2.5 - Debt Forgiveness",
    "E.3.2.6 - Debt Restructuring",
    "E.3.2.7 - Concessional Financing",
    "E.3.2.8 - Tied Aid Provision",
    "E.3.2.9 - Debt Sustainability Attack",
    "E.3.2.10 - Currency Swap Dependency Creation",
    "E.3.3.1 - Opaque Contract Terms",
    "E.3.3.2 - Cross-Default Provisions",
    "E.3.3.3 - Collateralized Sovereignty",
    "E.3.3.4 - Choice of Law Manipulation",
    "E.3.3.5 - Escrow Account Control",
    "E.3.3.6 - Commercial Rate Penalties",

    # E.4
    "E.4.1.1 - Market Access Denial",
    "E.4.1.2 - Discriminatory Treatment",
    "E.4.1.3 - Regulatory Harassment",
    "E.4.1.4 - Standards Manipulation",
    "E.4.1.5 - Customs Delay/Obstruction",
    "E.4.1.6 - Boycott Organization",
    "E.4.1.7 - Consumer Campaign",
    "E.4.1.8 - Regulatory Uncertainty Creation",
    "E.4.2.1 - Supply Chain Disruption",
    "E.4.2.2 - Critical Input Restriction",
    "E.4.2.3 - Rare Earth Export Control",
    "E.4.2.4 - Energy Supply Cutoff",
    "E.4.2.5 - Food Export Ban",
    "E.4.2.6 - Medical Supply Restriction",
    "E.4.2.7 - Transportation Blockade",
    "E.4.2.8 - Just-in-Time Supply Chain Disruption",
    "E.4.3.1 - Critical Mineral Processing Monopoly",
    "E.4.3.2 - API/Pharmaceutical Ingredient Control",
    "E.4.3.3 - Container Availability Manipulation",
    "E.4.3.4 - Port Terminal Prioritization",
    "E.4.3.5 - Shipping Insurance Denial",
    "E.4.3.6 - Strategic Component Withholding",

    # E.5
    "E.5.1.1 - Economic Union Formation",
    "E.5.1.2 - Currency Union Creation",
    "E.5.1.3 - Single Market Establishment",
    "E.5.1.4 - Economic Partnership Agreement",
    "E.5.1.5 - Regional Economic Community",
    "E.5.1.6 - Economic Corridor Development",
    "E.5.2.1 - Joint Venture Establishment",
    "E.5.2.2 - Technology Transfer Agreement",
    "E.5.2.3 - Industrial Park Development",
    "E.5.2.4 - Supply Chain Integration",
    "E.5.2.5 - Standards Harmonization",
    "E.5.2.6 - Joint R&D Program",
    "E.5.2.7 - Standard Setting Alliance",
    "E.5.2.8 - Digital Currency Cooperation",

    # E.6
    "E.6.1.1 - Energy Export Ban",
    "E.6.1.2 - Pipeline Construction/Closure",
    "E.6.1.3 - Energy Transit Denial",
    "E.6.1.4 - Energy Price Manipulation",
    "E.6.1.5 - Long-term Energy Contract",
    "E.6.1.6 - Energy Infrastructure Attack",
    "E.6.1.7 - Strategic Reserve Release",
    "E.6.1.8 - Strategic Reserve Manipulation",
    "E.6.2.1 - Resource Nationalization",
    "E.6.2.2 - Mining Rights Grant/Revocation",
    "E.6.2.3 - Fishing Rights Dispute",
    "E.6.2.4 - Water Resource Control",
    "E.6.2.5 - Agricultural Land Acquisition",
    "E.6.2.6 - Timber/Forestry Restriction",
    "E.6.2.7 - Carbon Credit Weaponization",

    # E.7
    "E.7.1.1 - Supply Chain Reshoring Initiative",
    "E.7.1.2 - Friend-shoring Policy",
    "E.7.1.3 - Near-shoring Program",
    "E.7.1.4 - Strategic Autonomy Initiative",
    "E.7.1.5 - Technology Sovereignty Program",
    "E.7.1.6 - Selective Decoupling",
    "E.7.1.7 - Economic Bloc Formation",
    "E.7.1.8 - Parallel Supply Chain Creation",
    "E.7.2.1 - Economic Integration Reversal",
    "E.7.2.2 - Dependency Reduction Program",
    "E.7.2.3 - Critical Mineral Independence",

    # F.1
    "F.1.1.1 - Currency Manipulation",
    "F.1.1.2 - Competitive Devaluation",
    "F.1.1.3 - Currency Attack/Speculation",
    "F.1.1.4 - Foreign Exchange Control",
    "F.1.1.5 - Currency Swap Agreement",
    "F.1.1.6 - Dollar/Euro Weaponization",
    "F.1.1.7 - De-dollarization Initiative",
    "F.1.1.8 - Digital Currency Launch",
    "F.1.1.9 - Synthetic Currency Creation",
    "F.1.1.10 - Currency Peg Attack",
    "F.1.2.1 - Interest Rate Manipulation",
    "F.1.2.2 - Quantitative Easing/Tightening",
    "F.1.2.3 - Foreign Reserve Freezing",
    "F.1.2.4 - Gold Reserve Repatriation",
    "F.1.2.5 - Central Bank Swap Line",
    "F.1.2.6 - Reserve Currency Status Change",
    "F.1.2.7 - Coordinated Central Bank Action",

    # F.2
    "F.2.1.1 - SWIFT Disconnection",
    "F.2.1.2 - Correspondent Banking Cutoff",
    "F.2.1.3 - Bank Account Closure",
    "F.2.1.4 - Transaction Blocking",
    "F.2.1.5 - Wire Transfer Prohibition",
    "F.2.1.6 - Credit Card Network Ban",
    "F.2.1.7 - Banking License Revocation",
    "F.2.1.8 - De-Risking Pressure",
    "F.2.2.1 - Alternative Payment Network Creation",
    "F.2.2.2 - Bilateral Payment Agreement",
    "F.2.2.3 - Cryptocurrency Adoption",
    "F.2.2.4 - Barter Trade Agreement",
    "F.2.2.5 - Local Currency Trade Agreement",
    "F.2.2.6 - Payment System Integration",
    "F.2.2.7 - Parallel Banking System Creation",
    "F.2.2.8 - CBDC Cross-Border Payment System",
    "F.2.2.9 - Digital Currency Diplomacy",

    # F.3
    "F.3.1.1 - Capital Market Ban",
    "F.3.1.2 - IPO Blocking",
    "F.3.1.3 - Bond Market Exclusion",
    "F.3.1.4 - Stock Market Delisting",
    "F.3.1.5 - Derivatives Trading Ban",
    "F.3.1.6 - Investment Ban",
    "F.3.1.7 - Sovereign Bond Attack",
    "F.3.2.1 - Stock Market Manipulation",
    "F.3.2.2 - Bond Market Manipulation",
    "F.3.2.3 - Commodity Market Manipulation",
    "F.3.2.4 - Short Selling Attack",
    "F.3.2.5 - Credit Rating Manipulation",
    "F.3.2.6 - Market Rumor Campaign",
    "F.3.2.7 - Algorithmic Market Manipulation",
    "F.3.2.8 - Flash Crash Triggering",

    # F.4
    "F.4.1.1 - IMF Program Influence",
    "F.4.1.2 - World Bank Loan Blocking",
    "F.4.1.3 - Regional Development Bank Control",
    "F.4.1.4 - Multilateral Loan Condition",
    "F.4.1.5 - Voting Share Manipulation",
    "F.4.1.6 - New Development Bank Creation",
    "F.4.2.1 - Sovereign Debt Purchase",
    "F.4.2.2 - Debt Default Trigger",
    "F.4.2.3 - Vulture Fund Operation",
    "F.4.2.4 - Debt-for-Equity Swap",
    "F.4.2.5 - Debt Restructuring Pressure",
    "F.4.2.6 - Paris Club Negotiation",
    "F.4.2.7 - Sovereign Wealth Fund Weaponization",
    "F.4.2.8 - Development Finance Competition",

    # F.5
    "F.5.1.1 - AML Designation",
    "F.5.1.2 - Financial Action Task Force Listing",
    "F.5.1.3 - Know Your Customer Restriction",
    "F.5.1.4 - Suspicious Activity Reporting",
    "F.5.1.5 - Financial Intelligence Sharing",
    "F.5.2.1 - Asset Seizure/Forfeiture",
    "F.5.2.2 - Unexplained Wealth Order",
    "F.5.2.3 - Beneficial Ownership Disclosure",
    "F.5.2.4 - Shell Company Crackdown",
    "F.5.2.5 - Tax Haven Blacklisting",
    "F.5.2.6 - Selective Enforcement Campaign",

    # F.6
    "F.6.1.1 - Digital Payment Platform Ban",
    "F.6.1.2 - Mobile Money Restriction",
    "F.6.1.3 - Blockchain Sanction Evasion",
    "F.6.1.4 - DeFi Platform Targeting",
    "F.6.1.5 - Stablecoin Regulation",
    "F.6.1.6 - NFT/Digital Asset Sanction Evasion",
    "F.6.2.1 - Financial Data Center Attack",
    "F.6.2.2 - Trading System Disruption",
    "F.6.2.3 - Financial Network Infiltration",
    "F.6.2.4 - High-Frequency Trading Attack",
    "F.6.2.5 - Quantum Computing Threat to Financial Cryptography",

    # IN.1
    "IN.1.1.1 - Agent Recruitment",
    "IN.1.1.2 - Agent Insertion/Infiltration",
    "IN.1.1.3 - Deep Cover Operation",
    "IN.1.1.4 - Double Agent Operation",
    "IN.1.1.5 - Asset Defection",
    "IN.1.1.6 - Walk-in Recruitment",
    "IN.1.1.7 - Honeytrap Operation",
    "IN.1.1.8 - False Flag Recruitment",
    "IN.1.1.9 - Academic/Researcher Recruitment",
    "IN.1.1.10 - Insider Threat Cultivation Through Social Media",
    "IN.1.2.1 - Diplomatic Cover Operation",
    "IN.1.2.2 - Embassy Intelligence Station",
    "IN.1.2.3 - Commercial Cover Operation",
    "IN.1.2.4 - NOC (Non-Official Cover) Deployment",
    "IN.1.2.5 - Intelligence Liaison Exchange",
    "IN.1.2.6 - Third-Country Operation",
    "IN.1.2.7 - Talent Program Recruitment",
    "IN.1.2.8 - Joint Laboratory Exploitation",

    # IN.2
    "IN.2.1.1 - Diplomatic Cable Interception",
    "IN.2.1.2 - Military Communication Interception",
    "IN.2.1.3 - Cellular Network Surveillance",
    "IN.2.1.4 - Internet Traffic Monitoring",
    "IN.2.1.5 - Satellite Communication Interception",
    "IN.2.1.6 - Undersea Cable Tapping",
    "IN.2.1.7 - Bulk Data Collection",
    "IN.2.1.8 - Encrypted Application Compromise",
    "IN.2.2.1 - Radar Signature Collection",
    "IN.2.2.2 - Electronic Order of Battle Mapping",
    "IN.2.2.3 - Telemetry Intelligence",
    "IN.2.2.4 - Foreign Instrumentation Signals",
    "IN.2.2.5 - Electromagnetic Pulse Detection",
    "IN.2.2.6 - 5G Network Exploitation",

    # IN.3
    "IN.3.1.1 - APT (Advanced Persistent Threat) Campaign",
    "IN.3.1.2 - Zero-Day Exploit Deployment",
    "IN.3.1.3 - Supply Chain Compromise",
    "IN.3.1.4 - Watering Hole Attack",
    "IN.3.1.5 - Spear Phishing Campaign",
    "IN.3.1.6 - Insider Threat Activation",
    "IN.3.1.7 - IoT Device Compromise",
    "IN.3.1.8 - Machine Learning Model Poisoning",
    "IN.3.2.1 - Mass Surveillance Program",
    "IN.3.2.2 - Targeted Device Compromise",
    "IN.3.2.3 - Cloud Service Infiltration",
    "IN.3.2.4 - Social Media Monitoring",
    "IN.3.2.5 - Encrypted Communication Breaking",
    "IN.3.2.6 - Tor/Dark Web Monitoring",
    "IN.3.2.7 - Behavioral Analytics Collection",

    # IN.4
    "IN.4.1.1 - Satellite Reconnaissance",
    "IN.4.1.2 - Aerial Reconnaissance",
    "IN.4.1.3 - Drone Surveillance",
    "IN.4.1.4 - Ground-Based Photography",
    "IN.4.1.5 - Commercial Imagery Purchase",
    "IN.4.1.6 - Synthetic Aperture Radar",
    "IN.4.1.7 - Hyperspectral Imaging",
    "IN.4.1.8 - Commercial Space Asset Exploitation",
    "IN.4.2.1 - Nuclear Test Detection",
    "IN.4.2.2 - Missile Launch Detection",
    "IN.4.2.3 - Chemical Signature Analysis",
    "IN.4.2.4 - Acoustic Intelligence",
    "IN.4.2.5 - Seismic Intelligence",
    "IN.4.2.6 - Materials Sampling",
    "IN.4.2.7 - Biometric Signature Collection",

    # IN.5
    "IN.5.1.1 - Mole Hunt Operation",
    "IN.5.1.2 - Security Clearance Investigation",
    "IN.5.1.3 - Polygraph Screening",
    "IN.5.1.4 - Technical Surveillance Countermeasures",
    "IN.5.1.5 - Communications Security Audit",
    "IN.5.1.6 - Insider Threat Program",
    "IN.5.2.1 - Double Agent Running",
    "IN.5.2.2 - Deception Operation",
    "IN.5.2.3 - Disinformation Feeding",
    "IN.5.2.4 - Foreign Intelligence Disruption",
    "IN.5.2.5 - Agent Provocation",
    "IN.5.2.6 - Intelligence Service Penetration",
    "IN.5.2.7 - Honeypot Infrastructure Creation",
    "IN.5.2.8 - False Intelligence Feed Operation",

    # IN.6
    "IN.6.1.1 - Election Interference",
    "IN.6.1.2 - Political Party Funding",
    "IN.6.1.3 - Opposition Support",
    "IN.6.1.4 - Protest Organization/Funding",
    "IN.6.1.5 - Civil Society Infiltration",
    "IN.6.1.6 - Media Outlet Control",
    "IN.6.1.7 - Agent of Influence Operation",
    "IN.6.1.8 - Corruption Network Creation",
    "IN.6.2.1 - Assassination/Targeted Killing",
    "IN.6.2.2 - Kidnapping/Rendition",
    "IN.6.2.3 - Sabotage Operation",
    "IN.6.2.4 - Arms Smuggling",
    "IN.6.2.5 - Training Camp Operation",
    "IN.6.2.6 - False Flag Attack",
    "IN.6.2.7 - Prison Break Operation",
    "IN.6.2.8 - Deniable Critical Infrastructure Attack",

    # IN.7
    "IN.7.1.1 - Five Eyes Intelligence Sharing",
    "IN.7.1.2 - Bilateral Intelligence Agreement",
    "IN.7.1.3 - Multilateral Intelligence Fusion",
    "IN.7.1.4 - Real-Time Intelligence Feed",
    "IN.7.1.5 - Database Access Agreement",
    "IN.7.1.6 - Joint Intelligence Center",
    "IN.7.2.1 - Joint Surveillance Operation",
    "IN.7.2.2 - Joint Cyber Operation",
    "IN.7.2.3 - Joint Counter-Terrorism Operation",
    "IN.7.2.4 - Joint Counter-Intelligence Operation",
    "IN.7.2.5 - Intelligence Capacity Building",
    "IN.7.2.6 - Commercial Intelligence Fusion",

    # IN.8
    "IN.8.1.1 - False Flag Attribution",
    "IN.8.1.2 - Multi-Actor Confusion",
    "IN.8.1.3 - Cutout Chain Operation",
    "IN.8.1.4 - Time-Delayed Attribution",
    "IN.8.1.5 - Attribution Flooding",
    "IN.8.2.1 - Preemptive Attribution",
    "IN.8.2.2 - Attribution Narrative Control",
    "IN.8.2.3 - Technical Indicator Manipulation",

    # L.1
    "L.1.1.1 - Transnational Crime Network Disruption",
    "L.1.1.2 - Drug Trafficking Interdiction",
    "L.1.1.3 - Human Trafficking Operation",
    "L.1.1.4 - Arms Trafficking Interdiction",
    "L.1.1.5 - Money Laundering Investigation",
    "L.1.1.6 - Cybercrime Investigation",
    "L.1.1.7 - Wildlife Trafficking Enforcement",
    "L.1.1.8 - Sanctions Evasion Network",
    "L.1.1.9 - Cryptocurrency Crime Investigation",
    "L.1.2.1 - Counter-Terrorism Operation",
    "L.1.2.2 - Terrorist Financing Investigation",
    "L.1.2.3 - Foreign Fighter Interdiction",
    "L.1.2.4 - Terrorist Designation",
    "L.1.2.5 - Terror Network Disruption",
    "L.1.2.6 - Radicalization Prevention",

    # L.2
    "L.2.1.1 - Extradition Request",
    "L.2.1.2 - Mutual Legal Assistance",
    "L.2.1.3 - Joint Investigation Team",
    "L.2.1.4 - Law Enforcement Liaison Officer",
    "L.2.1.5 - Evidence Sharing Agreement",
    "L.2.1.6 - Witness Protection Cooperation",
    "L.2.1.7 - Selective Cooperation Withdrawal",
    "L.2.2.1 - INTERPOL Red Notice",
    "L.2.2.2 - INTERPOL Blue Notice",
    "L.2.2.3 - Europol Joint Operation",
    "L.2.2.4 - Regional Police Cooperation",
    "L.2.2.5 - International Task Force",
    "L.2.2.6 - Capacity Building Program",

    # L.3
    "L.3.1.1 - Border Closure",
    "L.3.1.2 - Border Wall/Fence Construction",
    "L.3.1.3 - Border Patrol Enhancement",
    "L.3.1.4 - Biometric System Deployment",
    "L.3.1.5 - Smart Border Technology",
    "L.3.1.6 - Cross-Border Hot Pursuit",
    "L.3.2.1 - Mass Deportation",
    "L.3.2.2 - Visa Restriction/Ban",
    "L.3.2.3 - Refugee Pushback",
    "L.3.2.4 - Safe Third Country Agreement",
    "L.3.2.5 - Readmission Agreement",
    "L.3.2.6 - Immigration Raid",
    "L.3.2.7 - Weaponized Migration",
    "L.3.2.8 - Citizenship Stripping",

    # L.4
    "L.4.1.1 - Port Security Operation",
    "L.4.1.2 - Container Inspection Program",
    "L.4.1.3 - Ship Boarding/Inspection",
    "L.4.1.4 - Maritime Domain Awareness",
    "L.4.1.5 - Illegal Fishing Enforcement",
    "L.4.1.6 - Marine Environmental Enforcement",
    "L.4.2.1 - Search and Rescue",
    "L.4.2.2 - Drug Interdiction at Sea",
    "L.4.2.3 - Migrant Interdiction",
    "L.4.2.4 - Exclusive Economic Zone Patrol",
    "L.4.2.5 - Maritime Law Enforcement Agreement",
    "L.4.2.6 - Para-Naval Law Enforcement",
    "L.4.2.7 - Fishing Fleet Swarming",

    # L.5
    "L.5.1.1 - Universal Jurisdiction Claim",
    "L.5.1.2 - Extraterritorial Law Application",
    "L.5.1.3 - Long-Arm Statute Use",
    "L.5.1.4 - Maritime Law Exploitation",
    "L.5.1.5 - Space Law Claim",
    "L.5.1.6 - Cyber Jurisdiction Assertion",
    "L.5.1.7 - Retroactive Law Application",
    "L.5.2.1 - Political Prosecution",
    "L.5.2.2 - Selective Law Enforcement",
    "L.5.2.3 - Asset Freeze/Seizure",
    "L.5.2.4 - Travel Ban/Arrest Warrant",
    "L.5.2.5 - Corporate Criminal Prosecution",
    "L.5.2.6 - RICO/Conspiracy Charges",
    "L.5.2.7 - Strategic Litigation Funding",
    "L.5.3.1 - Fait Accompli Normalization",
    "L.5.3.2 - Customary Practice Creation",
    "L.5.3.3 - Effective Control Establishment",
    "L.5.3.4 - Prescriptive Rights Assertion",
    "L.5.3.5 - Adverse Possession Claim",
    "L.5.3.6 - Historical Rights Revival",
    "L.5.4.1 - Referendum Manipulation",
    "L.5.4.2 - Plebiscite Organization",
    "L.5.4.3 - Constitutional Crisis Creation",
    "L.5.4.4 - Recall Election Weaponization",
    "L.5.4.5 - Initiative/Ballot Measure Manipulation",
    "L.5.4.6 - No-Confidence Vote Engineering",

    # L.6
    "L.6.1.1 - Antitrust/Competition Enforcement",
    "L.6.1.2 - Foreign Corrupt Practices Act",
    "L.6.1.3 - Export Control Violation",
    "L.6.1.4 - Sanctions Violation Prosecution",
    "L.6.1.5 - Tax Evasion Prosecution",
    "L.6.1.6 - Regulatory Fine/Penalty",
    "L.6.2.1 - Data Privacy Enforcement",
    "L.6.2.2 - Cybersecurity Regulation",
    "L.6.2.3 - Encryption Regulation",
    "L.6.2.4 - Content Moderation Order",
    "L.6.2.5 - Platform Regulation",
    "L.6.2.6 - AI/Algorithm Regulation",
    "L.6.2.7 - Algorithm Audit Weaponization",
    "L.6.2.8 - Source Code Disclosure Requirement",

    # L.7
    "L.7.1.1 - International Criminal Court Referral",
    "L.7.1.2 - International Court of Justice Case",
    "L.7.1.3 - Regional Human Rights Court",
    "L.7.1.4 - International Arbitration",
    "L.7.1.5 - War Crimes Prosecution",
    "L.7.1.6 - Universal Jurisdiction Prosecution",
    "L.7.2.1 - Letters Rogatory",
    "L.7.2.2 - Evidence Collection Abroad",
    "L.7.2.3 - Witness Testimony Coordination",
    "L.7.2.4 - Asset Recovery Assistance",
    "L.7.2.5 - Prisoner Transfer Agreement",
    "L.7.2.6 - Joint Legal Task Force",
    "L.7.2.7 - Judicial Intimidation CampaignAdd to Conversation",

    # L.8
    "L.8.1.1 - Overseas Police Station Operation",
    "L.8.1.2 - Involuntary Return Operations (Fox Hunt)",
    "L.8.1.3 - Family Hostage Coercion",
    "L.8.1.4 - Covert Rendition Operations",
    "L.8.2.1 - Community Organization Capture",
    "L.8.2.2 - Digital Ecosystem Surveillance (WeChat)",
    "L.8.2.3 - Social Credit Extension Abroad",
    "L.8.2.4 - Exit Ban Leverage",

    # EN.1
    "EN.1.1.1 - Upstream Dam Flow Manipulation",
    "EN.1.1.2 - Sediment Trapping Operations",
    "EN.1.1.3 - Weaponized Water Release",
    "EN.1.1.4 - Water Quality Degradation",
    "EN.1.1.5 - Aquifer Contamination",
    "EN.1.1.6 - River Diversion Operations",

    # EN.2
    "EN.2.1.1 - Cloud Seeding Operations",
    "EN.2.1.2 - Rainfall Diversion",
    "EN.2.1.3 - Hurricane/Typhoon Path Influence",
    "EN.2.2.1 - Transboundary Acid Rain",
    "EN.2.2.2 - Strategic Smog Generation",
    "EN.2.2.3 - Toxic Waste Dumping",

    # EN.3
    "EN.3.1.1 - IUU Fishing Operations",
    "EN.3.1.2 - Breeding Ground Destruction",
    "EN.3.1.3 - Invasive Species Introduction",
    "EN.3.2.1 - Crop Disease Introduction",
    "EN.3.2.2 - Pollinator Disruption",
    "EN.3.2.3 - Soil Degradation Operations",
    "EN.3.2.4 - Seed Monopolization",
]

# ============================================================================
# Influence Pattern Tags
# ============================================================================
INFLUENCE_PATTERNS = [
    # Escalatory
    "escalation-spiral",
    "tit-for-tat",
    "provocation-response",
    # Containment
    "containment",
    "deterrence",
    "firebreak",
    # Position
    "power-transition",
    "consolidation",
    "erosion",
    "fait-accompli",
    # Coalition
    "alliance-formation",
    "alliance-dissolution",
    "wedge-driving",
    "bandwagoning",
    "balancing",
    # Signaling
    "signaling",
    "testing-probing",
    "demonstration",
    "normalization",
    # Maneuvering
    "diversion",
    "cumulative-gain",
    "ambiguous-action",
    "threshold-testing",
    # System State
    "homeostatic",
    "catalytic",
    "dampening",
    "amplifying",
]

# ============================================================================
# Type System Enums
# ============================================================================

ACTOR_POSTURES = [
    "assertive", "defensive", "cooperative", "neutral",
    "coercive", "provocative", "conciliatory", "withdrawn",
]

REGIONAL_TENSIONS = ["increasing", "decreasing", "stable", "volatile"]

RESOURCE_TYPES = [
    "military-assets", "diplomatic-capital", "economic-leverage",
    "political-capital", "intelligence-resources", "legal-standing",
    "reputation", "financial-instruments", "none-apparent",
]

COST_MAGNITUDES = ["negligible", "low", "moderate", "significant", "major"]

RISK_LEVELS = ["none", "low", "moderate", "high", "extreme"]

REVERSIBILITY_LEVELS = [
    "easily-reversible", "reversible", "difficult-to-reverse", "irreversible",
]

SURPRISE_ELEMENTS = ["none", "tactical", "strategic"]

CONSTANCY_INDICATORS = [
    "one-off", "repeated", "sustained-campaign", "escalating-series",
]

SITUATION_FRAMINGS = [
    "saving-face", "justice-penalty", "history",
    "sovereignty", "security", "economic-interest",
    "alliance-obligation", "humanitarian", "none-clear",
]

INITIATOR_ROLES = [
    "aggressor", "defender", "enforcer", "mediator",
    "provocateur", "protector", "claimant", "respondent",
]

TARGET_ROLES = [
    "victim", "violator", "competitor", "ally",
    "bystander", "subject", "challenger", "defender",
]

# ============================================================================
# Counterfactual / Implicit / Trade-off Enums
# ============================================================================

UNSTATED_OBJECTIVE_HINTS = [
    "capability-demonstration", "precedent-setting", "normalization",
    "option-creation", "option-foreclosure", "audience-signaling",
    "dependency-creation", "testing-boundaries",
    "deterrence", "salami-slicing", "fait-accompli",
    "none-apparent",
]

AUDIENCE_TARGETS = [
    "domestic", "domestic-public", "domestic-nationalist", "domestic-elite",
    "regional", "international",
    "adversary", "adversary-government", "adversary-military", "adversary-public",
    "ally", "neutral",
]

COST_OMITTED_HINTS = [
    "reputational", "alliance-trust", "escalation-risk",
    "legal-standing", "economic-blowback", "domestic-opposition",
    "precedent-risk", "none-apparent",
]

TIME_HORIZONS = ["immediate", "short-term", "medium-term", "long-term"]

GAIN_DURABILITIES = ["permanent", "durable", "fragile", "temporary"]

# ============================================================================
# Downstream enrichment enums (clustering, denoising, pattern mining)
# ============================================================================

ACTOR_SALIENCES = ["protagonist", "antagonist", "witness", "bystander"]

ACTION_SALIENCES = ["primary", "secondary", "background"]

RETALIATION_SIGNALS = ["explicit", "implied", "none"]

COORDINATION_SIGNALS = ["explicit", "implied", "none"]

NARRATIVE_ROLES = ["main_event", "enabling_context", "consequence", "background"]

CAUSAL_LINK_TYPES = ["causes", "enables", "prevents", "responds-to", "unknown"]

# ============================================================================
# Organisation types (from actor schema)
# ============================================================================
ORGANISATION_TYPES = [
    "government",
    "military",
    "intelligence",
    "law-enforcement",
    "legislature",
    "judiciary",
    "diplomatic",
    "state-media",
    "media",
    "civil-society",
    "private-sector",
    "armed-group",
    "individual",
]

# ============================================================================
# Location hierarchy types
# ============================================================================
LOCATION_LEVELS = [
    "city",             # Beijing, Tokyo, Canberra, Manila
    "admin_region",     # Xinjiang, Queensland, Okinawa, Crimea
    "country",          # When the location itself is a country
    "body_of_water",    # South China Sea, Indian Ocean, Taiwan Strait
    "strait",           # Taiwan Strait, Malacca Strait, Lombok Strait
    "island",           # Taiwan, Spratly Islands, Okinawa, Guam
    "reef_shoal",       # Scarborough Shoal, Mischief Reef, Second Thomas Shoal
    "military_base",    # Camp Humphreys, Pine Gap, Diego Garcia
    "port",             # Subic Bay, Cam Ranh Bay, Hambantota Port
    "border_zone",      # Line of Actual Control, DMZ, 38th Parallel
    "airspace",         # Taiwan ADIZ, South China Sea ADIZ
    "region",           # Southeast Asia, Horn of Africa, Korean Peninsula
    "other",
]

MACRO_REGIONS = [
    "East Asia",        # China, Japan, Korea, Taiwan, Mongolia
    "Southeast Asia",   # ASEAN states, Timor-Leste
    "South Asia",       # India, Pakistan, Bangladesh, Sri Lanka, Nepal
    "Central Asia",     # Kazakhstan, Uzbekistan, etc.
    "Middle East",      # Gulf states, Iran, Iraq, Israel, Turkey
    "Europe",           # EU, UK, Russia (western), Balkans
    "North America",    # US, Canada, Mexico
    "South America",    # Brazil, Argentina, etc.
    "Africa",           # All African states
    "Oceania",          # Australia, NZ, Pacific Islands
    "Arctic",           # Arctic region
    "Indo-Pacific",     # Broad cross-region (when spanning multiple)
    "Global",           # Global scope or multiple regions
]

# ============================================================================
# Edge relationship types (from influence_types.json)
# ============================================================================
EDGE_RELATION_TYPES = [
    # Causal influence
    "causes",
    "enables",
    "prevents",
    # Reactive
    "responds-to",
    # Amplifying/Modifying
    "reinforces",
    "contradicts",
    "supports",
    # Collaborative
    "coordinates-with",
    # Structural
    "instance-of",
    "reports-on",
    # Spatio-temporal
    "located-at",
    "occurred-on",
    # Actor-structural
    "initiates",
    "targets",
    "belongs-to",
    "represents",
    # Causal/temporal (Type System)
    "escalates-to",
    "de-escalates-to",
    "triggers",
    "retaliates-for",
    # State transitions
    "changes-state-of",
    "maintains-state-of",
]

STRENGTH_LEVELS = ["definite", "likely", "possible", "speculative"]

TEMPORAL_PRECISION = [
    "exact", "hour", "day", "week", "month", "year", "approximate", "relative-only",
]

# ============================================================================
# Type System enums (16-type dependent type system)
# ============================================================================

ACTION_INTENSITIES = ["low", "moderate", "high", "extreme"]

STATE_IMPACTS = ["status_quo", "incremental", "significant", "transformative"]

SURPRISE_LEVELS = ["expected", "unexpected", "ambiguous"]

AMBIGUITY_LEVELS = ["clear", "deniable", "ambiguous"]

INITIATION_TYPES = ["proactive", "reactive", "pre-positioned"]

NARRATIVE_FRAMINGS = [
    "sovereignty", "self-defense", "rules-based", "historical-claim",
    "humanitarian", "economic-rights", "freedom-of-navigation", "neutral",
]

# ============================================================================
# Category B: Structurally missing fields (from CustomerValidated gap analysis)
# ============================================================================

AUTHORITATIVENESS_LEVELS = [
    "head-of-state", "head-of-government", "minister", "senior-official",
    "military-commander", "spokesperson", "diplomat", "analyst",
    "state-media", "independent-media", "anonymous-source", "unknown",
]

RESOLUTION_PATHWAYS = [
    "negotiation", "mediation", "arbitration", "adjudication",
    "unilateral-withdrawal", "mutual-de-escalation", "status-quo-acceptance",
    "fait-accompli", "ongoing-contestation", "not-applicable",
]

COOPERATION_TYPES = [
    "military", "diplomatic", "economic", "intelligence",
    "law-enforcement", "humanitarian", "scientific", "multi-domain",
]

# ============================================================================
# Cross-document structural hints (improve dedup, clustering, nesting)
# ============================================================================

CONTINUITY_SIGNALS = [
    "new-event",              # first occurrence / novel event
    "continuation",           # ongoing series (e.g. "30th consecutive day")
    "escalation-of-prior",    # explicit escalation from earlier event
    "response-to-prior",      # reactive to a specific prior event
    "follow-up",              # update or development of known story
    "recurrence",             # similar event happened before
]

PERSPECTIVE_ALIGNMENTS = [
    "pro-initiator",   # article frames initiator favorably
    "pro-target",      # article frames target favorably
    "neutral",         # balanced or factual framing
    "mixed",           # multiple perspectives presented
    "third-party",     # framed from an outside observer's perspective
]

# ============================================================================
# NuExtract Template (semantic guidance)
# ============================================================================
# NuExtract-2.0 uses this template to understand field semantics.
# Array fields with string items = "extract all matching spans"
# Enum arrays = "pick one from this list"
# Nested objects = structured extraction

GRAYZONE_RELEVANCE = ["high", "medium", "low", "none"]

NUEXTRACT_TEMPLATE = """{
  "grayzone_relevance": """ + str(GRAYZONE_RELEVANCE) + """,
  "situation_label": "",
  "topic_keywords": [""],
  "actions": [
    {
      "action_description": "",
      "temporal_order": 0,
      "action_salience": """ + str(ACTION_SALIENCES) + """,
      "narrative_role": """ + str(NARRATIVE_ROLES) + """,
      "retaliation_signal": """ + str(RETALIATION_SIGNALS) + """,
      "coordination_signal": """ + str(COORDINATION_SIGNALS) + """,
      "l1_domain": """ + str(L1_DOMAINS) + """,
      "l3_code": """ + str(L3_SUBDOMAINS) + """,
      "escalation_classification": ["escalation", "de-escalation", "cooperation-building", "cooperation-maintaining", "neutral", "ambiguous"],
      "escalation_magnitude": 0.0,
      "influence_patterns": """ + str(INFLUENCE_PATTERNS) + """,
      "scope": ["bilateral", "regional", "global"],
      "claim_type": ["territorial-maritime", "territorial-land", "territorial-airspace", "economic-zone", "resource-rights", "legal-jurisdiction", "access-rights", "no-claims"],
      "contestation_dynamics": ["escalating-contestation", "maintaining-contestation", "reducing-contestation", "neutral-no-active-claims", "building-cooperation", "strengthening-alliance", "deepening-partnership", "maintaining-cooperation"],
      "initiators": [
        {
          "name": "",
          "canonical_name": "",
          "country_code": "",
          "organisation": "",
          "organisation_type": """ + str(ORGANISATION_TYPES) + """,
          "role": "",
          "individual_name": "",
          "salience": """ + str(ACTOR_SALIENCES) + """
        }
      ],
      "targets": [
        {
          "name": "",
          "canonical_name": "",
          "country_code": "",
          "organisation": "",
          "organisation_type": """ + str(ORGANISATION_TYPES) + """,
          "role": "",
          "individual_name": "",
          "salience": """ + str(ACTOR_SALIENCES) + """
        }
      ],
      "locations": [
        {
          "name": "",
          "canonical_name": "",
          "level": """ + str(LOCATION_LEVELS) + """,
          "country_code": "",
          "macro_region": """ + str(MACRO_REGIONS) + """
        }
      ],
      "date_raw": "",
      "date_iso": "",
      "date_end_raw": "",
      "date_end_iso": "",
      "date_precision": ["exact", "day", "week", "month", "year", "approximate", "relative-only"],
      "timezone": "",
      "relative_temporal": "",
      "assets": [""],
      "state_indicators": {
        "actor_posture": """ + str(ACTOR_POSTURES) + """,
        "regional_tension": """ + str(REGIONAL_TENSIONS) + """,
        "prior_state_description": "",
        "resulting_state_description": ""
      },
      "cost_signals": {
        "resource_type": """ + str(RESOURCE_TYPES) + """,
        "cost_magnitude": """ + str(COST_MAGNITUDES) + """,
        "risk_level": """ + str(RISK_LEVELS) + """,
        "reversibility": """ + str(REVERSIBILITY_LEVELS) + """
      },
      "asymmetry_signals": {
        "surprise_element": """ + str(SURPRISE_ELEMENTS) + """,
        "ambiguity_level": """ + str(AMBIGUITY_LEVELS) + """,
        "cross_domain_linkage": "",
        "constancy_indicator": """ + str(CONSTANCY_INDICATORS) + """
      },
      "situation_framing": """ + str(SITUATION_FRAMINGS) + """,
      "actor_role_framing": {
        "initiator_role": """ + str(INITIATOR_ROLES) + """,
        "target_role": """ + str(TARGET_ROLES) + """
      },
      "action_verb": "",
      "action_intensity": """ + str(ACTION_INTENSITIES) + """,
      "state_impact": """ + str(STATE_IMPACTS) + """,
      "narrative_framing": """ + str(NARRATIVE_FRAMINGS) + """,
      "situation_signals": {
        "saving_face_indicators": [""],
        "justice_penalty_indicators": [""],
        "historical_references": [""]
      },
      "decision_context": {
        "alternatives_mentioned": [""],
        "constraints_cited": [""],
        "preconditions": [""],
        "enabling_factors": [""],
        "conditional_threats": [""],
        "explicit_counterfactuals": [""]
      },
      "implicit_signals": {
        "stated_justification": "",
        "unstated_objective_hint": """ + str(UNSTATED_OBJECTIVE_HINTS) + """,
        "denial_or_deflection": "",
        "audience_targeting": """ + str(AUDIENCE_TARGETS) + """,
        "capability_revealed": "",
        "precedent_implications": ""
      },
      "trade_off_signals": {
        "benefit_claimed": "",
        "cost_acknowledged": "",
        "cost_omitted_hint": """ + str(COST_OMITTED_HINTS) + """,
        "time_horizon": """ + str(TIME_HORIZONS) + """,
        "dependency_created": "",
        "reversibility_of_gain": """ + str(GAIN_DURABILITIES) + """
      },
      "authoritativeness": """ + str(AUTHORITATIVENESS_LEVELS) + """,
      "resolution_pathway": """ + str(RESOLUTION_PATHWAYS) + """,
      "dimension_shift": false,
      "cooperation_type": """ + str(COOPERATION_TYPES) + """,
      "cooperation_partners": [""],
      "event_fingerprint": "",
      "continuity_signal": """ + str(CONTINUITY_SIGNALS) + """,
      "perspective_alignment": """ + str(PERSPECTIVE_ALIGNMENTS) + """,
      "related_event_references": [""]
    }
  ],
  "edges": [
    {
      "source_action_index": 0,
      "target_action_index": 1,
      "relation_type": """ + str(EDGE_RELATION_TYPES) + """,
      "causal_link_type": """ + str(CAUSAL_LINK_TYPES) + """,
      "strength": """ + str(STRENGTH_LEVELS) + """,
      "description": ""
    }
  ],
  "source_text": ""
}"""


# ============================================================================
# xgrammar JSON Schema (structural constraint)
# ============================================================================

_ACTOR_SCHEMA = {
    "type": "object",
    "required": ["name", "canonical_name", "country_code", "organisation_type", "salience"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "canonical_name": {"type": "string"},
        "country_code": {"type": "string", "minLength": 2, "maxLength": 3},
        "organisation": {"type": "string"},
        "organisation_type": {
            "type": "string",
            "enum": ORGANISATION_TYPES,
        },
        "role": {"type": "string"},
        "individual_name": {"type": "string"},
        "salience": {
            "type": "string",
            "enum": ACTOR_SALIENCES,
        },
    },
}

_LOCATION_SCHEMA = {
    "type": "object",
    "required": ["name", "canonical_name", "level", "macro_region"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "canonical_name": {"type": "string"},
        "level": {
            "type": "string",
            "enum": LOCATION_LEVELS,
        },
        "country_code": {"type": "string", "minLength": 2, "maxLength": 3},
        "macro_region": {
            "type": "string",
            "enum": MACRO_REGIONS,
        },
    },
}

XGRAMMAR_SCHEMA = {
    "type": "object",
    "required": ["grayzone_relevance", "actions", "source_text"],
    "additionalProperties": False,
    "properties": {
        "grayzone_relevance": {
            "type": "string",
            "enum": GRAYZONE_RELEVANCE,
        },
        "situation_label": {
            "type": "string",
            "description": "Broad conflict/situation name for clustering (e.g. 'Senkaku Islands dispute')",
        },
        "topic_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
            "description": "3-5 topical keywords for situation clustering and nesting",
        },
        "source_text": {
            "type": "string",
            "description": "Raw article text passthrough for downstream processing",
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "action_description", "temporal_order",
                    "action_salience", "narrative_role",
                    "retaliation_signal", "coordination_signal",
                    "l1_domain", "l3_code",
                    "escalation_classification", "escalation_magnitude",
                    "influence_patterns",
                    "initiators", "targets",
                ],
                "additionalProperties": False,
                "properties": {
                    "action_description": {"type": "string"},
                    "temporal_order": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "action_salience": {
                        "type": "string",
                        "enum": ACTION_SALIENCES,
                    },
                    "narrative_role": {
                        "type": "string",
                        "enum": NARRATIVE_ROLES,
                    },
                    "retaliation_signal": {
                        "type": "string",
                        "enum": RETALIATION_SIGNALS,
                    },
                    "coordination_signal": {
                        "type": "string",
                        "enum": COORDINATION_SIGNALS,
                    },
                    "l1_domain": {
                        "type": "string",
                        "enum": L1_DOMAINS,
                    },
                    "l3_code": {
                        "type": "string",
                        "enum": L3_SUBDOMAINS,
                    },
                    "escalation_classification": {
                        "type": "string",
                        "enum": [
                            "escalation", "de-escalation",
                            "cooperation-building", "cooperation-maintaining",
                            "neutral", "ambiguous",
                        ],
                    },
                    "escalation_magnitude": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "influence_patterns": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": INFLUENCE_PATTERNS,
                        },
                        "maxItems": 3,
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["bilateral", "regional", "global"],
                    },
                    "claim_type": {
                        "type": "string",
                        "enum": [
                            "territorial-maritime", "territorial-land",
                            "territorial-airspace", "economic-zone",
                            "resource-rights", "legal-jurisdiction",
                            "access-rights", "no-claims",
                        ],
                    },
                    "contestation_dynamics": {
                        "type": "string",
                        "enum": [
                            "escalating-contestation", "maintaining-contestation",
                            "reducing-contestation", "neutral-no-active-claims",
                            "building-cooperation", "strengthening-alliance",
                            "deepening-partnership", "maintaining-cooperation",
                        ],
                    },
                    "initiators": {
                        "type": "array",
                        "items": _ACTOR_SCHEMA,
                    },
                    "targets": {
                        "type": "array",
                        "items": _ACTOR_SCHEMA,
                    },
                    "locations": {
                        "type": "array",
                        "items": _LOCATION_SCHEMA,
                    },
                    "date_raw": {"type": "string"},
                    "date_iso": {"type": "string"},
                    "date_end_raw": {"type": "string"},
                    "date_end_iso": {"type": "string"},
                    "date_precision": {
                        "type": "string",
                        "enum": TEMPORAL_PRECISION,
                    },
                    "timezone": {"type": "string"},
                    "relative_temporal": {"type": "string"},
                    "assets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "state_indicators": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "actor_posture": {
                                "type": "string",
                                "enum": ACTOR_POSTURES,
                            },
                            "regional_tension": {
                                "type": "string",
                                "enum": REGIONAL_TENSIONS,
                            },
                            "prior_state_description": {"type": "string"},
                            "resulting_state_description": {"type": "string"},
                        },
                    },
                    "cost_signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "resource_type": {
                                "type": "string",
                                "enum": RESOURCE_TYPES,
                            },
                            "cost_magnitude": {
                                "type": "string",
                                "enum": COST_MAGNITUDES,
                            },
                            "risk_level": {
                                "type": "string",
                                "enum": RISK_LEVELS,
                            },
                            "reversibility": {
                                "type": "string",
                                "enum": REVERSIBILITY_LEVELS,
                            },
                        },
                    },
                    "asymmetry_signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "surprise_element": {
                                "type": "string",
                                "enum": SURPRISE_ELEMENTS,
                            },
                            "ambiguity_level": {
                                "type": "string",
                                "enum": AMBIGUITY_LEVELS,
                            },
                            "cross_domain_linkage": {"type": "string"},
                            "constancy_indicator": {
                                "type": "string",
                                "enum": CONSTANCY_INDICATORS,
                            },
                        },
                    },
                    "situation_framing": {
                        "type": "string",
                        "enum": SITUATION_FRAMINGS,
                    },
                    "actor_role_framing": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "initiator_role": {
                                "type": "string",
                                "enum": INITIATOR_ROLES,
                            },
                            "target_role": {
                                "type": "string",
                                "enum": TARGET_ROLES,
                            },
                        },
                    },
                    "action_verb": {"type": "string"},
                    "action_intensity": {
                        "type": "string",
                        "enum": ACTION_INTENSITIES,
                    },
                    "state_impact": {
                        "type": "string",
                        "enum": STATE_IMPACTS,
                    },
                    "narrative_framing": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": NARRATIVE_FRAMINGS,
                        },
                        "maxItems": 3,
                    },
                    "situation_signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "saving_face_indicators": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "justice_penalty_indicators": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "historical_references": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "decision_context": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "alternatives_mentioned": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "constraints_cited": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "preconditions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "enabling_factors": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "conditional_threats": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "explicit_counterfactuals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "implicit_signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "stated_justification": {"type": "string"},
                            "unstated_objective_hint": {
                                "type": "string",
                                "enum": UNSTATED_OBJECTIVE_HINTS,
                            },
                            "denial_or_deflection": {"type": "string"},
                            "audience_targeting": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": AUDIENCE_TARGETS,
                                },
                                "maxItems": 3,
                            },
                            "capability_revealed": {"type": "string"},
                            "precedent_implications": {"type": "string"},
                        },
                    },
                    "trade_off_signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "benefit_claimed": {"type": "string"},
                            "cost_acknowledged": {"type": "string"},
                            "cost_omitted_hint": {
                                "type": "string",
                                "enum": COST_OMITTED_HINTS,
                            },
                            "time_horizon": {
                                "type": "string",
                                "enum": TIME_HORIZONS,
                            },
                            "dependency_created": {"type": "string"},
                            "reversibility_of_gain": {
                                "type": "string",
                                "enum": GAIN_DURABILITIES,
                            },
                        },
                    },
                    # --- Category B: structurally missing fields ---
                    "authoritativeness": {
                        "type": "string",
                        "enum": AUTHORITATIVENESS_LEVELS,
                    },
                    "resolution_pathway": {
                        "type": "string",
                        "enum": RESOLUTION_PATHWAYS,
                    },
                    "dimension_shift": {"type": "boolean"},
                    "cooperation_type": {
                        "type": "string",
                        "enum": COOPERATION_TYPES,
                    },
                    "cooperation_partners": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    # --- Cross-document structural hints ---
                    "event_fingerprint": {
                        "type": "string",
                        "description": "Canonical ~50 char event description for cross-document dedup",
                    },
                    "continuity_signal": {
                        "type": "string",
                        "enum": CONTINUITY_SIGNALS,
                    },
                    "perspective_alignment": {
                        "type": "string",
                        "enum": PERSPECTIVE_ALIGNMENTS,
                    },
                    "related_event_references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "References to other events mentioned in article text",
                    },
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_action_index", "target_action_index", "relation_type"],
                "additionalProperties": False,
                "properties": {
                    "source_action_index": {"type": "integer", "minimum": 0},
                    "target_action_index": {"type": "integer", "minimum": 0},
                    "relation_type": {
                        "type": "string",
                        "enum": EDGE_RELATION_TYPES,
                    },
                    "causal_link_type": {
                        "type": "string",
                        "enum": CAUSAL_LINK_TYPES,
                    },
                    "strength": {
                        "type": "string",
                        "enum": STRENGTH_LEVELS,
                    },
                    "description": {"type": "string"},
                },
            },
        },
    },
}


# ============================================================================
# Few-shot examples for relevance calibration
# ============================================================================
# These examples teach the model what counts as grayzone-relevant vs not.
# vLLM prefix caching ensures the shared prefix is computed only once.

_EXAMPLE_IRRELEVANT_1 = (
    "Pakistan installs 500,000 toilets in rural Sindh province under a new "
    "public health initiative. The World Bank praised the sanitation program "
    "as a model for developing nations. Local officials said the project "
    "would reduce waterborne diseases by 40 percent."
)
_EXAMPLE_IRRELEVANT_1_OUTPUT = '{"grayzone_relevance": "none", "situation_label": "", "topic_keywords": [], "actions": [], "source_text": "Pakistan installs 500,000 toilets in rural Sindh province under a new public health initiative. The World Bank praised the sanitation program as a model for developing nations. Local officials said the project would reduce waterborne diseases by 40 percent."}'

_EXAMPLE_IRRELEVANT_2 = (
    "South Korea's Samsung Electronics reported record semiconductor sales "
    "in Q3, driven by strong demand for AI chips. The company plans to invest "
    "$30 billion in a new fab in Texas. Analysts expect continued growth in "
    "the memory chip market through 2026."
)
_EXAMPLE_IRRELEVANT_2_OUTPUT = '{"grayzone_relevance": "none", "situation_label": "", "topic_keywords": [], "actions": [], "source_text": "South Korea\'s Samsung Electronics reported record semiconductor sales in Q3, driven by strong demand for AI chips. The company plans to invest $30 billion in a new fab in Texas. Analysts expect continued growth in the memory chip market through 2026."}'

# Chinese input -> English output (teaches translate-and-extract pattern)
_EXAMPLE_CHINESE_IRRELEVANT = (
    "中国国家统计局公布数据显示，2024年第三季度国内生产总值同比增长4.6%。"
    "消费支出继续成为经济增长的主要动力，零售销售额同比增长5.2%。"
    "分析人士预计下半年经济将保持稳定增长态势。"
)
_EXAMPLE_CHINESE_IRRELEVANT_OUTPUT = '{"grayzone_relevance": "none", "situation_label": "", "topic_keywords": [], "actions": [], "source_text": "China\'s National Bureau of Statistics released data showing GDP grew 4.6% year-on-year in Q3 2024. Consumer spending continued to be the main driver of economic growth, with retail sales up 5.2% year-on-year. Analysts expect steady growth in the second half of the year."}'

_EXAMPLE_CHINESE_RELEVANT = (
    "中国海警局发言人表示，中国海警舰艇编队当日在钓鱼岛领海内巡航。"
    "日本海上保安厅称已派出巡视船跟踪监视。外交部发言人回应称，"
    "钓鱼岛是中国固有领土，中方在本国领海巡逻是正当合法的。"
    "日本外务省随后向中方提出严正抗议。"
)
_EXAMPLE_CHINESE_RELEVANT_OUTPUT = '{"grayzone_relevance": "high", "situation_label": "Diaoyu/Senkaku Islands patrol dispute", "topic_keywords": ["Diaoyu Islands", "China Coast Guard", "Japan Coast Guard", "maritime patrol", "territorial dispute"], "actions": [{"action_description": "China Coast Guard fleet conducted patrol in Diaoyu Islands territorial waters", "temporal_order": 0, "action_salience": "primary", "narrative_role": "main_event", "retaliation_signal": "none", "coordination_signal": "none", "l1_domain": "M", "l3_code": "Maritime Gray Zone Harassment", "escalation_classification": "confrontational", "escalation_magnitude": 0.6, "influence_patterns": ["escalation-spiral"], "scope": "bilateral", "claim_type": "territorial-maritime", "contestation_dynamics": "escalating-contestation", "initiators": [{"name": "China Coast Guard", "canonical_name": "China Coast Guard", "country_code": "CN", "organisation": "China Coast Guard", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "protagonist"}], "targets": [{"name": "Japan", "canonical_name": "Japan", "country_code": "JP", "organisation": "Government of Japan", "organisation_type": "government", "role": "claimant", "individual_name": "", "salience": "antagonist"}], "locations": [{"name": "Diaoyu Islands", "canonical_name": "Senkaku Islands", "level": "island", "country_code": "JP", "macro_region": "East Asia"}], "date_raw": "", "date_iso": "", "date_precision": "approximate", "assets": ["coast guard vessels"], "state_indicators": {"actor_posture": "assertive", "regional_tension": "elevated", "prior_state_description": "", "resulting_state_description": ""}, "cost_signals": {"resource_type": "military-assets", "cost_magnitude": "moderate", "risk_level": "moderate", "reversibility": "reversible"}, "asymmetry_signals": {"surprise_element": "none", "ambiguity_level": "clear", "cross_domain_linkage": "", "constancy_indicator": "sustained-campaign"}, "situation_framing": "sovereignty", "actor_role_framing": {"initiator_role": "assertive-claimant", "target_role": "defender"}, "action_verb": "patrolled", "action_intensity": "moderate", "state_impact": "incremental", "narrative_framing": ["sovereignty"], "situation_signals": {"saving_face_indicators": [], "justice_penalty_indicators": [], "historical_references": []}, "decision_context": {"alternatives_mentioned": [], "constraints_cited": [], "preconditions": [], "enabling_factors": [], "conditional_threats": [], "explicit_counterfactuals": []}, "implicit_signals": {"stated_justification": "inherent territory", "unstated_objective_hint": "salami-slicing", "denial_or_deflection": "", "audience_targeting": ["domestic-nationalist"], "capability_revealed": "", "precedent_implications": ""}, "trade_off_signals": {"benefit_claimed": "", "cost_acknowledged": "", "cost_omitted_hint": "none-apparent", "time_horizon": "medium-term", "dependency_created": "", "reversibility_of_gain": "fragile"}, "date_end_raw": "", "date_end_iso": "", "timezone": "Asia/Tokyo", "relative_temporal": "", "authoritativeness": "senior-official", "resolution_pathway": "ongoing-contestation", "dimension_shift": false, "cooperation_type": "", "cooperation_partners": [], "event_fingerprint": "CCG patrol in Diaoyu/Senkaku territorial waters", "continuity_signal": "continuation", "perspective_alignment": "pro-initiator", "related_event_references": []}, {"action_description": "Japan Coast Guard dispatched patrol boats to track and monitor Chinese vessels", "temporal_order": 1, "action_salience": "secondary", "narrative_role": "response", "retaliation_signal": "implied", "coordination_signal": "none", "l1_domain": "M", "l3_code": "Naval Operations", "escalation_classification": "confrontational", "escalation_magnitude": 0.5, "influence_patterns": ["tit-for-tat"], "scope": "bilateral", "claim_type": "territorial-maritime", "contestation_dynamics": "escalating-contestation", "initiators": [{"name": "Japan Coast Guard", "canonical_name": "Japan Coast Guard", "country_code": "JP", "organisation": "Japan Coast Guard", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "protagonist"}], "targets": [{"name": "China Coast Guard", "canonical_name": "China Coast Guard", "country_code": "CN", "organisation": "China Coast Guard", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "antagonist"}], "locations": [{"name": "Diaoyu Islands", "canonical_name": "Senkaku Islands", "level": "island", "country_code": "JP", "macro_region": "East Asia"}], "date_raw": "", "date_iso": "", "date_precision": "approximate", "assets": ["patrol boats"], "state_indicators": {"actor_posture": "defensive", "regional_tension": "elevated", "prior_state_description": "", "resulting_state_description": ""}, "cost_signals": {"resource_type": "military-assets", "cost_magnitude": "moderate", "risk_level": "moderate", "reversibility": "reversible"}, "asymmetry_signals": {"surprise_element": "none", "ambiguity_level": "clear", "cross_domain_linkage": "", "constancy_indicator": "repeated"}, "situation_framing": "sovereignty", "actor_role_framing": {"initiator_role": "defender", "target_role": "challenger"}, "action_verb": "dispatched", "action_intensity": "moderate", "state_impact": "incremental", "narrative_framing": ["sovereignty"], "situation_signals": {"saving_face_indicators": [], "justice_penalty_indicators": [], "historical_references": []}, "decision_context": {"alternatives_mentioned": [], "constraints_cited": [], "preconditions": [], "enabling_factors": [], "conditional_threats": [], "explicit_counterfactuals": []}, "implicit_signals": {"stated_justification": "", "unstated_objective_hint": "deterrence", "denial_or_deflection": "", "audience_targeting": ["domestic-public"], "capability_revealed": "", "precedent_implications": ""}, "trade_off_signals": {"benefit_claimed": "", "cost_acknowledged": "", "cost_omitted_hint": "none-apparent", "time_horizon": "short-term", "dependency_created": "", "reversibility_of_gain": "fragile"}, "date_end_raw": "", "date_end_iso": "", "timezone": "Asia/Tokyo", "relative_temporal": "", "authoritativeness": "senior-official", "resolution_pathway": "ongoing-contestation", "dimension_shift": false, "cooperation_type": "", "cooperation_partners": [], "event_fingerprint": "JCG dispatches patrol boats to monitor CCG near Senkaku", "continuity_signal": "response-to-prior", "perspective_alignment": "pro-initiator", "related_event_references": ["CCG patrol in Diaoyu/Senkaku territorial waters"]}], "edges": [{"source_action_index": 0, "target_action_index": 1, "relation_type": "causes", "causal_link_type": "causes", "strength": "definite", "description": "CCG patrol directly prompted JCG monitoring response"}], "source_text": "A China Coast Guard spokesperson stated that a CCG fleet conducted a patrol within the territorial waters of the Diaoyu Islands. Japan Coast Guard said it dispatched patrol boats to track and monitor the vessels. A Chinese Foreign Ministry spokesperson responded that the Diaoyu Islands are inherent Chinese territory and that patrols in Chinese territorial waters are legitimate and lawful. Japan Ministry of Foreign Affairs subsequently lodged a formal protest with the Chinese side."}'

_EXAMPLE_RELEVANT = (
    "China Coast Guard vessels entered waters near the Senkaku Islands for "
    "the 30th consecutive day, prompting Japan to file a diplomatic protest. "
    "Tokyo summoned Beijing's ambassador and demanded an immediate withdrawal. "
    "The Japanese Maritime Self-Defense Force dispatched two destroyers to "
    "monitor the situation."
)
_EXAMPLE_RELEVANT_OUTPUT = '{"grayzone_relevance": "high", "situation_label": "Senkaku Islands maritime dispute", "topic_keywords": ["Senkaku Islands", "China Coast Guard", "maritime incursion", "Japan diplomatic protest", "JMSDF"], "actions": [{"action_description": "China Coast Guard vessels entered waters near the Senkaku Islands for the 30th consecutive day", "temporal_order": 0, "action_salience": "primary", "narrative_role": "main_event", "retaliation_signal": "none", "coordination_signal": "none", "l1_domain": "M", "l3_code": "Maritime Gray Zone Harassment", "escalation_classification": "escalation", "escalation_magnitude": 0.7, "influence_patterns": ["escalation-spiral"], "scope": "bilateral", "claim_type": "territorial-maritime", "contestation_dynamics": "escalating-contestation", "initiators": [{"name": "China Coast Guard", "canonical_name": "China Coast Guard", "country_code": "CN", "organisation": "China Coast Guard", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "protagonist"}], "targets": [{"name": "Japan", "canonical_name": "Japan", "country_code": "JP", "organisation": "Government of Japan", "organisation_type": "government", "role": "claimant", "individual_name": "", "salience": "antagonist"}], "locations": [{"name": "Senkaku Islands", "canonical_name": "Senkaku Islands", "level": "island", "country_code": "JP", "macro_region": "East Asia"}], "date_raw": "", "date_iso": "", "date_precision": "approximate", "assets": ["coast guard vessels"], "state_indicators": {"actor_posture": "assertive", "regional_tension": "elevated", "prior_state_description": "", "resulting_state_description": ""}, "cost_signals": {"resource_type": "military-assets", "cost_magnitude": "moderate", "risk_level": "moderate", "reversibility": "reversible"}, "asymmetry_signals": {"surprise_element": "none", "ambiguity_level": "clear", "cross_domain_linkage": "", "constancy_indicator": "sustained-campaign"}, "situation_framing": "sovereignty", "actor_role_framing": {"initiator_role": "provocateur", "target_role": "defender"}, "action_verb": "entered", "action_intensity": "moderate", "state_impact": "incremental", "narrative_framing": ["sovereignty"], "situation_signals": {"saving_face_indicators": [], "justice_penalty_indicators": [], "historical_references": []}, "decision_context": {"alternatives_mentioned": [], "constraints_cited": [], "preconditions": [], "enabling_factors": [], "conditional_threats": [], "explicit_counterfactuals": []}, "implicit_signals": {"stated_justification": "", "unstated_objective_hint": "salami-slicing", "denial_or_deflection": "", "audience_targeting": ["domestic-nationalist"], "capability_revealed": "", "precedent_implications": ""}, "trade_off_signals": {"benefit_claimed": "", "cost_acknowledged": "", "cost_omitted_hint": "none-apparent", "time_horizon": "medium-term", "dependency_created": "", "reversibility_of_gain": "fragile"}, "date_end_raw": "", "date_end_iso": "", "timezone": "Asia/Tokyo", "relative_temporal": "for the 30th consecutive day", "authoritativeness": "independent-media", "resolution_pathway": "ongoing-contestation", "dimension_shift": false, "cooperation_type": "", "cooperation_partners": [], "event_fingerprint": "CCG vessels enter Senkaku waters, 30th consecutive day", "continuity_signal": "continuation", "perspective_alignment": "pro-target", "related_event_references": []}, {"action_description": "Japan filed a diplomatic protest and summoned Beijing’s ambassador demanding immediate withdrawal", "temporal_order": 1, "action_salience": "secondary", "narrative_role": "response", "retaliation_signal": "explicit", "coordination_signal": "none", "l1_domain": "D", "l3_code": "Formal Protests", "escalation_classification": "escalation", "escalation_magnitude": 0.4, "influence_patterns": ["tit-for-tat"], "scope": "bilateral", "claim_type": "territorial-maritime", "contestation_dynamics": "escalating-contestation", "initiators": [{"name": "Japan", "canonical_name": "Japan", "country_code": "JP", "organisation": "Government of Japan", "organisation_type": "government", "role": "claimant", "individual_name": "", "salience": "protagonist"}], "targets": [{"name": "China", "canonical_name": "China", "country_code": "CN", "organisation": "Government of China", "organisation_type": "government", "role": "challenger", "individual_name": "", "salience": "antagonist"}], "locations": [{"name": "Tokyo", "canonical_name": "Tokyo", "level": "city", "country_code": "JP", "macro_region": "East Asia"}], "date_raw": "", "date_iso": "", "date_precision": "approximate", "assets": [], "state_indicators": {"actor_posture": "assertive", "regional_tension": "elevated", "prior_state_description": "", "resulting_state_description": ""}, "cost_signals": {"resource_type": "diplomatic-capital", "cost_magnitude": "low", "risk_level": "low", "reversibility": "easily-reversible"}, "asymmetry_signals": {"surprise_element": "none", "ambiguity_level": "clear", "cross_domain_linkage": "", "constancy_indicator": "repeated"}, "situation_framing": "sovereignty", "actor_role_framing": {"initiator_role": "defender", "target_role": "challenger"}, "action_verb": "protested", "action_intensity": "moderate", "state_impact": "incremental", "narrative_framing": ["sovereignty"], "situation_signals": {"saving_face_indicators": [], "justice_penalty_indicators": [], "historical_references": []}, "decision_context": {"alternatives_mentioned": [], "constraints_cited": [], "preconditions": [], "enabling_factors": [], "conditional_threats": ["demanded immediate withdrawal"], "explicit_counterfactuals": []}, "implicit_signals": {"stated_justification": "", "unstated_objective_hint": "none-apparent", "denial_or_deflection": "", "audience_targeting": ["domestic-public"], "capability_revealed": "", "precedent_implications": ""}, "trade_off_signals": {"benefit_claimed": "", "cost_acknowledged": "", "cost_omitted_hint": "none-apparent", "time_horizon": "short-term", "dependency_created": "", "reversibility_of_gain": "fragile"}, "date_end_raw": "", "date_end_iso": "", "timezone": "Asia/Tokyo", "relative_temporal": "", "authoritativeness": "senior-official", "resolution_pathway": "negotiation", "dimension_shift": false, "cooperation_type": "", "cooperation_partners": [], "event_fingerprint": "Japan summons Chinese ambassador over Senkaku incursion", "continuity_signal": "response-to-prior", "perspective_alignment": "pro-initiator", "related_event_references": ["CCG 30th consecutive day Senkaku incursion"]}, {"action_description": "Japanese Maritime Self-Defense Force dispatched two destroyers to monitor the situation", "temporal_order": 2, "action_salience": "secondary", "narrative_role": "response", "retaliation_signal": "implied", "coordination_signal": "none", "l1_domain": "M", "l3_code": "Naval Operations", "escalation_classification": "escalation", "escalation_magnitude": 0.8, "influence_patterns": ["escalation-spiral"], "scope": "bilateral", "claim_type": "territorial-maritime", "contestation_dynamics": "escalating-contestation", "initiators": [{"name": "JMSDF", "canonical_name": "Japan Maritime Self-Defense Force", "country_code": "JP", "organisation": "Japan Maritime Self-Defense Force", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "protagonist"}], "targets": [{"name": "China Coast Guard", "canonical_name": "China Coast Guard", "country_code": "CN", "organisation": "China Coast Guard", "organisation_type": "military", "role": "patrol", "individual_name": "", "salience": "antagonist"}], "locations": [{"name": "Senkaku Islands", "canonical_name": "Senkaku Islands", "level": "island", "country_code": "JP", "macro_region": "East Asia"}], "date_raw": "", "date_iso": "", "date_precision": "approximate", "assets": ["destroyers"], "state_indicators": {"actor_posture": "assertive", "regional_tension": "elevated", "prior_state_description": "", "resulting_state_description": ""}, "cost_signals": {"resource_type": "military-assets", "cost_magnitude": "significant", "risk_level": "moderate", "reversibility": "reversible"}, "asymmetry_signals": {"surprise_element": "none", "ambiguity_level": "clear", "cross_domain_linkage": "diplomatic-to-military", "constancy_indicator": "repeated"}, "situation_framing": "sovereignty", "actor_role_framing": {"initiator_role": "defender", "target_role": "challenger"}, "action_verb": "dispatched", "action_intensity": "high", "state_impact": "incremental", "narrative_framing": ["sovereignty"], "situation_signals": {"saving_face_indicators": [], "justice_penalty_indicators": [], "historical_references": []}, "decision_context": {"alternatives_mentioned": [], "constraints_cited": [], "preconditions": [], "enabling_factors": [], "conditional_threats": [], "explicit_counterfactuals": []}, "implicit_signals": {"stated_justification": "", "unstated_objective_hint": "deterrence", "denial_or_deflection": "", "audience_targeting": ["domestic-public", "adversary-government"], "capability_revealed": "naval-deployment-readiness", "precedent_implications": ""}, "trade_off_signals": {"benefit_claimed": "", "cost_acknowledged": "", "cost_omitted_hint": "none-apparent", "time_horizon": "short-term", "dependency_created": "", "reversibility_of_gain": "fragile"}, "date_end_raw": "", "date_end_iso": "", "timezone": "Asia/Tokyo", "relative_temporal": "", "authoritativeness": "senior-official", "resolution_pathway": "ongoing-contestation", "dimension_shift": true, "cooperation_type": "", "cooperation_partners": [], "event_fingerprint": "JMSDF dispatches destroyers to Senkaku Islands", "continuity_signal": "escalation-of-prior", "perspective_alignment": "pro-initiator", "related_event_references": ["CCG 30th consecutive day Senkaku incursion", "Japan diplomatic protest"]}], "edges": [{"source_action_index": 0, "target_action_index": 1, "relation_type": "causes", "causal_link_type": "causes", "strength": "definite", "description": "CCG incursion directly caused Japan diplomatic protest"}, {"source_action_index": 0, "target_action_index": 2, "relation_type": "causes", "causal_link_type": "causes", "strength": "definite", "description": "CCG incursion prompted JMSDF naval deployment"}, {"source_action_index": 1, "target_action_index": 2, "relation_type": "enables", "causal_link_type": "enables", "strength": "likely", "description": "Diplomatic protest escalation path enabled military response"}], "source_text": "China Coast Guard vessels entered waters near the Senkaku Islands for the 30th consecutive day, prompting Japan to file a diplomatic protest. Tokyo summoned Beijing’s ambassador and demanded an immediate withdrawal. The Japanese Maritime Self-Defense Force dispatched two destroyers to monitor the situation."}'


# ============================================================================
# Message builder
# ============================================================================

def build_messages(article_text: str) -> list:
    """Build chat messages with few-shot examples for relevance calibration.

    Returns a list of message dicts for vLLM's llm.chat() interface.
    The NuExtract template is injected via chat_template_kwargs.
    vLLM prefix caching ensures the shared few-shot prefix is computed once.
    """
    return [
        {"role": "user", "content": _EXAMPLE_IRRELEVANT_1},
        {"role": "assistant", "content": _EXAMPLE_IRRELEVANT_1_OUTPUT},
        {"role": "user", "content": _EXAMPLE_IRRELEVANT_2},
        {"role": "assistant", "content": _EXAMPLE_IRRELEVANT_2_OUTPUT},
        {"role": "user", "content": _EXAMPLE_CHINESE_IRRELEVANT},
        {"role": "assistant", "content": _EXAMPLE_CHINESE_IRRELEVANT_OUTPUT},
        {"role": "user", "content": _EXAMPLE_CHINESE_RELEVANT},
        {"role": "assistant", "content": _EXAMPLE_CHINESE_RELEVANT_OUTPUT},
        {"role": "user", "content": _EXAMPLE_RELEVANT},
        {"role": "assistant", "content": _EXAMPLE_RELEVANT_OUTPUT},
        {"role": "user", "content": article_text},
    ]


# ============================================================================
# Convenience: print template stats
# ============================================================================

if __name__ == "__main__":
    import json

    print("=== Enhanced NuExtract Template (v2 + Type System) ===")
    print(f"L1 domains:         {len(L1_DOMAINS)}")
    print(f"L2 sub-domains:     {len(L2_SUBDOMAINS)}")
    print(f"Influence patterns:  {len(INFLUENCE_PATTERNS)}")
    print(f"Organisation types:  {len(ORGANISATION_TYPES)}")
    print(f"Edge relations:      {len(EDGE_RELATION_TYPES)}")
    print(f"Strength levels:     {len(STRENGTH_LEVELS)}")
    print(f"Temporal precision:  {len(TEMPORAL_PRECISION)}")
    print(f"Action intensities:  {len(ACTION_INTENSITIES)}")
    print(f"State impacts:       {len(STATE_IMPACTS)}")
    print(f"Narrative framings:  {len(NARRATIVE_FRAMINGS)}")
    print(f"Actor postures:      {len(ACTOR_POSTURES)}")
    print(f"Regional tensions:   {len(REGIONAL_TENSIONS)}")
    print(f"Resource types:      {len(RESOURCE_TYPES)}")
    print(f"Cost magnitudes:     {len(COST_MAGNITUDES)}")
    print(f"Risk levels:         {len(RISK_LEVELS)}")
    print(f"Reversibility lvls:  {len(REVERSIBILITY_LEVELS)}")
    print(f"Surprise elements:   {len(SURPRISE_ELEMENTS)}")
    print(f"Ambiguity levels:    {len(AMBIGUITY_LEVELS)}")
    print(f"Constancy indicators:{len(CONSTANCY_INDICATORS)}")
    print(f"Situation framings:  {len(SITUATION_FRAMINGS)}")
    print(f"Initiator roles:     {len(INITIATOR_ROLES)}")
    print(f"Target roles:        {len(TARGET_ROLES)}")
    print()
    print("Template length:", len(NUEXTRACT_TEMPLATE), "chars")
    print("Schema keys:", list(XGRAMMAR_SCHEMA["properties"].keys()))
    print()
    print("--- NuExtract Template ---")
    print(NUEXTRACT_TEMPLATE[:500] + "...")
    print()
    print("--- xgrammar Schema (pretty) ---")
    print(json.dumps(XGRAMMAR_SCHEMA, indent=2)[:1000] + "...")
