"""Generate a cognition AnchorSchema from the DIMEFIL taxonomy + geo hierarchy.

Produces a JSON file with:
  - DIMEFIL L1 → L2 → L3 relations as typed anchors with parent links
  - Location hierarchy: macro_region → country → key cities/features
  - Actor hierarchy: organisation_type → key state actors
  - Temporal precision anchors

Each anchor carries tokens (surface forms for SPLADE projection),
type, parent, and optional metadata (country_code, macro_region, level).

Usage:
    python cognition/generate_dimefil_schema.py
    python cognition/generate_dimefil_schema.py --output data/dimefil_schema.json
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# DIMEFIL relation hierarchy (L1 → L2 → L3)
# Only L3 gets tokens — L1/L2 are structural parents for rollup.
# L4 is too granular for SPLADE token projection (829 items), so we stop at L3
# and let the NLI head handle fine-grained classification.
# ---------------------------------------------------------------------------

DIMEFIL_TREE = {
    "diplomatic": {
        "type": "relation",
        "tokens": ["diplomatic", "diplomacy", "diplomat"],
        "children": {
            "bilateral_relations": {
                "tokens": ["bilateral", "relations"],
                "children": {
                    "state_visits": {"tokens": ["visit", "summit", "meeting", "delegation"]},
                    "diplomatic_recognition": {"tokens": ["recognition", "recognize", "recognise", "sever"]},
                    "embassy_operations": {"tokens": ["embassy", "consulate", "ambassador", "consular"]},
                    "bilateral_agreements": {"tokens": ["treaty", "agreement", "pact", "memorandum", "ratification"]},
                },
            },
            "multilateral_engagement": {
                "tokens": ["multilateral", "international organization"],
                "children": {
                    "intl_organizations": {"tokens": ["united nations", "un", "security council", "general assembly"]},
                    "regional_organizations": {"tokens": ["asean", "nato", "eu", "african union", "aukus", "quad"]},
                    "multilateral_treaties": {"tokens": ["multilateral treaty", "arms control", "non-proliferation"]},
                    "procedural_warfare": {"tokens": ["procedural", "agenda", "accreditation", "veto"]},
                },
            },
            "diplomatic_pressure": {
                "tokens": ["protest", "condemn", "pressure"],
                "children": {
                    "formal_protests": {"tokens": ["protest", "condemnation", "demarche", "summon", "boycott"]},
                    "diplomatic_isolation": {"tokens": ["isolation", "quarantine", "exclusion", "travel ban"]},
                },
            },
            "peace_mediation": {
                "tokens": ["peace", "mediation", "ceasefire"],
                "children": {
                    "conflict_resolution": {"tokens": ["peace process", "ceasefire", "armistice", "peace treaty"]},
                    "mediation_facilitation": {"tokens": ["mediation", "mediator", "arbitration", "shuttle diplomacy"]},
                },
            },
            "normalization_ops": {
                "tokens": ["normalization", "threshold", "legitimization"],
                "children": {
                    "threshold_shifting": {"tokens": ["threshold", "erosion", "precedent", "overton"]},
                    "legitimization": {"tokens": ["legitimization", "justification", "reinterpretation"]},
                },
            },
        },
    },
    "informational": {
        "type": "relation",
        "tokens": ["information", "informational", "media", "propaganda"],
        "children": {
            "strategic_comms": {
                "tokens": ["strategic communication", "public diplomacy", "messaging"],
                "children": {
                    "public_diplomacy": {"tokens": ["broadcasting", "cultural center", "exchange program", "branding"]},
                    "government_messaging": {"tokens": ["government statement", "press conference", "white paper", "announcement"]},
                },
            },
            "info_operations": {
                "tokens": ["information operation", "influence operation"],
                "children": {
                    "propaganda_influence": {"tokens": ["propaganda", "influence", "bot", "astroturfing", "troll"]},
                    "disinformation": {"tokens": ["disinformation", "deepfake", "false flag", "forgery", "conspiracy"]},
                    "info_warfare": {"tokens": ["cognitive warfare", "psyop", "narrative warfare", "memetic"]},
                },
            },
            "cyber_info_ops": {
                "tokens": ["cyber information", "hack", "leak"],
                "children": {
                    "cyber_enabled_info": {"tokens": ["defacement", "hijacking", "hack and release", "doxxing"]},
                    "info_control": {"tokens": ["internet shutdown", "censorship", "blocking", "throttling", "vpn"]},
                    "leak_ecosystem": {"tokens": ["leak", "whistleblower", "journalist asset", "timed release"]},
                },
            },
            "media_journalism": {
                "tokens": ["media", "journalism", "press"],
                "children": {
                    "media_relations": {"tokens": ["interview", "press embargo", "accreditation", "correspondent"]},
                    "media_control": {"tokens": ["media closure", "censorship", "editorial control", "journalist arrest"]},
                },
            },
        },
    },
    "military": {
        "type": "relation",
        "tokens": ["military", "armed forces", "defense", "defence"],
        "children": {
            "force_projection": {
                "tokens": ["force projection", "combat", "strike", "operation"],
                "children": {
                    "combat_operations": {"tokens": ["invasion", "strike", "bombardment", "bombing", "missile", "drone", "raid"]},
                    "military_posturing": {"tokens": ["exercise", "drill", "mobilization", "weapons test", "parade"]},
                    "naval_operations": {"tokens": ["naval", "blockade", "carrier", "submarine", "escort", "mine"]},
                    "air_operations": {"tokens": ["no-fly zone", "air patrol", "reconnaissance", "airlift", "bomber"]},
                    "maritime_gray_zone": {"tokens": ["coast guard", "harassment", "ramming", "fishing fleet", "militia"]},
                },
            },
            "military_presence": {
                "tokens": ["military base", "deployment", "basing"],
                "children": {
                    "forward_deployment": {"tokens": ["base establishment", "forward operating", "troop rotation", "advisor"]},
                    "base_operations": {"tokens": ["base expansion", "base closure", "port access", "overflight", "logistics hub"]},
                },
            },
            "defense_cooperation": {
                "tokens": ["defense cooperation", "military assistance", "alliance"],
                "children": {
                    "military_assistance": {"tokens": ["military sales", "military financing", "training", "equipping"]},
                    "alliance_operations": {"tokens": ["alliance", "collective defense", "article 5", "burden sharing"]},
                },
            },
            "unconventional_warfare": {
                "tokens": ["unconventional", "proxy", "irregular", "guerrilla"],
                "children": {
                    "proxy_operations": {"tokens": ["proxy", "militia", "mercenary", "private military", "foreign fighter"]},
                    "irregular_warfare": {"tokens": ["guerrilla", "insurgency", "sabotage", "subversion", "coup"]},
                },
            },
            "wmd_strategic": {
                "tokens": ["nuclear", "wmd", "missile", "chemical", "biological"],
                "children": {
                    "nuclear_operations": {"tokens": ["nuclear test", "nuclear alert", "nuclear doctrine", "nuclear umbrella"]},
                    "missile_operations": {"tokens": ["ballistic missile", "cruise missile", "hypersonic", "anti-satellite"]},
                    "cbrn_operations": {"tokens": ["chemical weapon", "biological weapon", "radiological"]},
                },
            },
            "space_cyber_mil": {
                "tokens": ["space military", "cyber military"],
                "children": {
                    "space_operations": {"tokens": ["satellite", "anti-satellite", "space weapon", "orbital"]},
                    "cyber_mil_operations": {"tokens": ["offensive cyber", "cyber espionage", "cyber attack", "cyber implant"]},
                },
            },
            "escalation_mgmt": {
                "tokens": ["escalation", "de-escalation"],
                "children": {
                    "escalation_control": {"tokens": ["escalation signal", "escalation pause", "proportional response"]},
                    "deescalation_ops": {"tokens": ["off-ramp", "step-back", "confidence building"]},
                },
            },
        },
    },
    "economic": {
        "type": "relation",
        "tokens": ["economic", "economy", "trade"],
        "children": {
            "trade_policy": {
                "tokens": ["trade policy", "tariff", "trade"],
                "children": {
                    "tariffs_duties": {"tokens": ["tariff", "duty", "customs", "anti-dumping", "countervailing"]},
                    "trade_agreements": {"tokens": ["free trade", "trade agreement", "customs union", "common market"]},
                    "import_export_controls": {"tokens": ["export ban", "import ban", "quota", "licensing", "export control"]},
                },
            },
            "economic_sanctions": {
                "tokens": ["sanction", "embargo"],
                "children": {
                    "comprehensive_sanctions": {"tokens": ["embargo", "economic embargo", "arms embargo", "comprehensive sanction"]},
                    "targeted_sanctions": {"tokens": ["targeted sanction", "asset freeze", "entity list", "secondary sanction"]},
                    "export_controls": {"tokens": ["dual-use", "technology control", "export restriction"]},
                    "humanitarian_econ_warfare": {"tokens": ["humanitarian exemption", "food security", "aid diversion"]},
                },
            },
            "investment_capital": {
                "tokens": ["investment", "capital", "fdi"],
                "children": {
                    "fdi": {"tokens": ["foreign direct investment", "fdi", "sovereign wealth", "nationalization", "expropriation"]},
                    "development_finance": {"tokens": ["development loan", "infrastructure investment", "belt and road", "debt trap"]},
                    "sovereign_debt_weapon": {"tokens": ["sovereign debt", "collateral", "cross-default"]},
                },
            },
            "economic_coercion": {
                "tokens": ["economic coercion", "market access"],
                "children": {
                    "market_access": {"tokens": ["market access denial", "discriminatory", "regulatory harassment", "boycott"]},
                    "supply_chain": {"tokens": ["supply chain", "critical input", "rare earth", "energy cutoff"]},
                    "chokepoint_control": {"tokens": ["chokepoint", "critical mineral", "shipping insurance", "port terminal"]},
                },
            },
            "economic_cooperation": {
                "tokens": ["economic cooperation", "economic integration"],
                "children": {
                    "economic_integration": {"tokens": ["economic union", "currency union", "single market", "economic corridor"]},
                    "industrial_cooperation": {"tokens": ["joint venture", "technology transfer", "standards harmonization"]},
                },
            },
            "resource_energy": {
                "tokens": ["resource", "energy", "pipeline"],
                "children": {
                    "energy_policy": {"tokens": ["energy export", "pipeline", "energy price", "strategic reserve"]},
                    "resource_control": {"tokens": ["nationalization", "mining rights", "fishing rights", "water resource"]},
                },
            },
            "economic_restructuring": {
                "tokens": ["decoupling", "reshoring", "friend-shoring"],
                "children": {
                    "decoupling_ops": {"tokens": ["reshoring", "friend-shoring", "near-shoring", "strategic autonomy", "decoupling"]},
                    "recoupling_ops": {"tokens": ["integration reversal", "dependency reduction", "critical mineral independence"]},
                },
            },
        },
    },
    "financial": {
        "type": "relation",
        "tokens": ["financial", "finance", "banking", "monetary"],
        "children": {
            "monetary_policy": {
                "tokens": ["monetary policy", "currency", "central bank"],
                "children": {
                    "currency_operations": {"tokens": ["currency manipulation", "devaluation", "de-dollarization", "digital currency"]},
                    "central_bank_actions": {"tokens": ["interest rate", "quantitative easing", "reserve freeze", "swap line"]},
                },
            },
            "banking_payments": {
                "tokens": ["banking", "payment", "swift"],
                "children": {
                    "banking_restrictions": {"tokens": ["swift", "correspondent banking", "transaction blocking", "wire transfer"]},
                    "alt_payment_systems": {"tokens": ["alternative payment", "cryptocurrency", "barter", "local currency", "cbdc"]},
                },
            },
            "capital_markets": {
                "tokens": ["capital market", "stock market", "bond"],
                "children": {
                    "financial_market_access": {"tokens": ["capital market ban", "ipo blocking", "delisting", "bond market"]},
                    "market_manipulation": {"tokens": ["market manipulation", "short selling", "credit rating", "flash crash"]},
                },
            },
            "intl_finance": {
                "tokens": ["international finance", "imf", "world bank"],
                "children": {
                    "multilateral_finance": {"tokens": ["imf", "world bank", "development bank", "multilateral loan"]},
                    "debt_operations": {"tokens": ["sovereign debt", "debt default", "vulture fund", "debt restructuring"]},
                },
            },
            "financial_crime": {
                "tokens": ["money laundering", "financial crime"],
                "children": {
                    "aml": {"tokens": ["anti-money laundering", "aml", "fatf", "kyc", "suspicious activity"]},
                    "asset_recovery": {"tokens": ["asset seizure", "forfeiture", "unexplained wealth", "shell company"]},
                },
            },
            "fintech_disruption": {
                "tokens": ["fintech", "digital payment", "defi"],
                "children": {
                    "fintech_weapon": {"tokens": ["payment platform ban", "mobile money", "blockchain", "defi", "stablecoin"]},
                    "financial_infra": {"tokens": ["financial data center", "trading system", "high-frequency trading"]},
                },
            },
        },
    },
    "intelligence": {
        "type": "relation",
        "tokens": ["intelligence", "espionage", "spy"],
        "children": {
            "humint": {
                "tokens": ["humint", "human intelligence", "agent"],
                "children": {
                    "agent_operations": {"tokens": ["agent recruitment", "infiltration", "deep cover", "double agent", "honeytrap"]},
                    "diplomatic_intelligence": {"tokens": ["diplomatic cover", "intelligence station", "commercial cover", "noc"]},
                },
            },
            "sigint": {
                "tokens": ["sigint", "signals intelligence", "interception"],
                "children": {
                    "comms_intelligence": {"tokens": ["cable interception", "cellular surveillance", "internet monitoring", "satellite interception"]},
                    "electronic_intelligence": {"tokens": ["radar signature", "telemetry", "electronic intelligence"]},
                },
            },
            "cyber_intelligence": {
                "tokens": ["cyber intelligence", "cyber espionage"],
                "children": {
                    "cyber_espionage": {"tokens": ["apt", "zero-day", "supply chain compromise", "spear phishing"]},
                    "cyber_surveillance": {"tokens": ["mass surveillance", "device compromise", "cloud infiltration", "dark web"]},
                },
            },
            "tech_intelligence": {
                "tokens": ["technical intelligence", "imagery intelligence"],
                "children": {
                    "imint": {"tokens": ["satellite reconnaissance", "aerial reconnaissance", "drone surveillance", "synthetic aperture"]},
                    "masint": {"tokens": ["nuclear test detection", "missile launch detection", "chemical signature", "acoustic intelligence"]},
                },
            },
            "counterintelligence": {
                "tokens": ["counterintelligence", "mole hunt"],
                "children": {
                    "defensive_ci": {"tokens": ["mole hunt", "security clearance", "polygraph", "insider threat"]},
                    "offensive_ci": {"tokens": ["double agent", "deception operation", "disinformation feed", "honeypot"]},
                },
            },
            "covert_action": {
                "tokens": ["covert action", "covert operation"],
                "children": {
                    "political_action": {"tokens": ["election interference", "political party funding", "opposition support", "protest funding"]},
                    "paramilitary_action": {"tokens": ["assassination", "rendition", "sabotage", "arms smuggling", "false flag"]},
                },
            },
            "intel_cooperation": {
                "tokens": ["intelligence sharing", "five eyes"],
                "children": {
                    "intel_sharing": {"tokens": ["five eyes", "intelligence agreement", "intelligence fusion", "database access"]},
                    "joint_intel_ops": {"tokens": ["joint surveillance", "joint cyber", "counter-terrorism", "counter-intelligence"]},
                },
            },
            "attribution_warfare": {
                "tokens": ["attribution", "false flag"],
                "children": {
                    "attribution_obfuscation": {"tokens": ["false flag attribution", "multi-actor confusion", "cutout"]},
                    "attribution_shaping": {"tokens": ["preemptive attribution", "attribution narrative", "technical indicator"]},
                },
            },
        },
    },
    "law_enforcement": {
        "type": "relation",
        "tokens": ["law enforcement", "police", "legal"],
        "children": {
            "transnational_crime": {
                "tokens": ["transnational crime", "organized crime", "trafficking"],
                "children": {
                    "organized_crime": {"tokens": ["organized crime", "drug trafficking", "human trafficking", "arms trafficking", "money laundering"]},
                    "terrorism": {"tokens": ["terrorism", "counter-terrorism", "terrorist financing", "radicalization"]},
                },
            },
            "intl_law_coop": {
                "tokens": ["extradition", "mutual legal assistance"],
                "children": {
                    "bilateral_law_coop": {"tokens": ["extradition", "mutual legal assistance", "joint investigation"]},
                    "multilateral_law_coop": {"tokens": ["interpol", "europol", "red notice", "international task force"]},
                },
            },
            "border_immigration": {
                "tokens": ["border", "immigration", "deportation"],
                "children": {
                    "border_control": {"tokens": ["border closure", "border wall", "border patrol", "biometric"]},
                    "immigration_enforcement": {"tokens": ["deportation", "visa restriction", "refugee", "weaponized migration"]},
                },
            },
            "maritime_law": {
                "tokens": ["maritime law", "coast guard", "port security"],
                "children": {
                    "maritime_security": {"tokens": ["port security", "container inspection", "ship boarding", "fishing enforcement"]},
                    "coast_guard_ops": {"tokens": ["search and rescue", "drug interdiction", "migrant interdiction", "eez patrol"]},
                },
            },
            "lawfare": {
                "tokens": ["lawfare", "legal warfare", "jurisdiction"],
                "children": {
                    "jurisdictional_assertion": {"tokens": ["universal jurisdiction", "extraterritorial", "long-arm statute"]},
                    "legal_persecution": {"tokens": ["political prosecution", "selective enforcement", "asset seizure"]},
                    "temporal_legal_ops": {"tokens": ["fait accompli", "customary practice", "adverse possession"]},
                    "democratic_process_weapon": {"tokens": ["referendum", "plebiscite", "constitutional crisis", "no-confidence"]},
                },
            },
            "regulatory_enforcement": {
                "tokens": ["regulatory", "antitrust", "compliance"],
                "children": {
                    "economic_regulation": {"tokens": ["antitrust", "competition enforcement", "sanctions violation", "tax evasion"]},
                    "technology_regulation": {"tokens": ["data privacy", "cybersecurity regulation", "encryption", "platform regulation", "ai regulation"]},
                },
            },
            "judicial_cooperation": {
                "tokens": ["judicial cooperation", "international court"],
                "children": {
                    "intl_courts": {"tokens": ["icc", "icj", "international criminal court", "war crimes", "arbitration"]},
                    "legal_assistance": {"tokens": ["letters rogatory", "evidence collection", "witness testimony", "prisoner transfer"]},
                },
            },
            "transnational_repression": {
                "tokens": ["transnational repression", "overseas police"],
                "children": {
                    "extraterritorial_law": {"tokens": ["overseas police station", "involuntary return", "fox hunt", "rendition"]},
                    "diaspora_control": {"tokens": ["diaspora", "community organization", "wechat surveillance", "social credit", "exit ban"]},
                },
            },
        },
    },
    "environmental": {
        "type": "relation",
        "tokens": ["environmental", "ecological"],
        "children": {
            "water_warfare": {
                "tokens": ["water warfare", "dam", "river"],
                "children": {
                    "hydro_hegemony": {"tokens": ["dam manipulation", "sediment trapping", "weaponized water", "aquifer", "river diversion"]},
                },
            },
            "atmospheric_ops": {
                "tokens": ["weather modification", "pollution"],
                "children": {
                    "weather_modification": {"tokens": ["cloud seeding", "rainfall diversion", "weather modification"]},
                    "pollution_export": {"tokens": ["acid rain", "toxic waste", "smog"]},
                },
            },
            "bio_resource_warfare": {
                "tokens": ["fisheries", "agricultural warfare"],
                "children": {
                    "fisheries_depletion": {"tokens": ["iuu fishing", "illegal fishing", "breeding ground", "invasive species"]},
                    "agricultural_warfare": {"tokens": ["crop disease", "pollinator", "soil degradation", "seed monopoly"]},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Location hierarchy: macro_region → country → key features
# ---------------------------------------------------------------------------

LOCATION_TREE = {
    "east_asia": {
        "tokens": ["east asia"],
        "macro_region": "East Asia",
        "children": {
            "china": {"tokens": ["china", "chinese", "beijing", "prc"], "country_code": "CN", "children": {
                "south_china_sea": {"tokens": ["south china sea", "scs"], "level": "body_of_water"},
                "taiwan_strait": {"tokens": ["taiwan strait"], "level": "strait"},
                "xinjiang": {"tokens": ["xinjiang", "uyghur"], "level": "admin_region"},
                "hong_kong": {"tokens": ["hong kong"], "level": "admin_region"},
                "tibet": {"tokens": ["tibet", "tibetan"], "level": "admin_region"},
                "hainan": {"tokens": ["hainan"], "level": "admin_region"},
            }},
            "japan": {"tokens": ["japan", "japanese", "tokyo"], "country_code": "JP", "children": {
                "senkaku": {"tokens": ["senkaku", "diaoyu"], "level": "island"},
                "okinawa": {"tokens": ["okinawa"], "level": "island"},
            }},
            "south_korea": {"tokens": ["south korea", "korean", "seoul", "rok"], "country_code": "KR", "children": {
                "dmz_korea": {"tokens": ["dmz", "demilitarized zone", "38th parallel"], "level": "border_zone"},
            }},
            "north_korea": {"tokens": ["north korea", "pyongyang", "dprk"], "country_code": "KP"},
            "taiwan": {"tokens": ["taiwan", "taipei", "formosa", "roc"], "country_code": "TW"},
            "mongolia": {"tokens": ["mongolia", "ulaanbaatar"], "country_code": "MN"},
        },
    },
    "southeast_asia": {
        "tokens": ["southeast asia", "asean"],
        "macro_region": "Southeast Asia",
        "children": {
            "philippines": {"tokens": ["philippines", "filipino", "manila"], "country_code": "PH", "children": {
                "scarborough_shoal": {"tokens": ["scarborough shoal"], "level": "reef_shoal"},
                "second_thomas_shoal": {"tokens": ["second thomas shoal", "ayungin"], "level": "reef_shoal"},
                "spratly_islands": {"tokens": ["spratly", "spratlys"], "level": "island"},
                "subic_bay": {"tokens": ["subic bay"], "level": "port"},
            }},
            "vietnam": {"tokens": ["vietnam", "vietnamese", "hanoi"], "country_code": "VN", "children": {
                "cam_ranh_bay": {"tokens": ["cam ranh bay"], "level": "port"},
                "paracel_islands": {"tokens": ["paracel", "paracels", "hoang sa"], "level": "island"},
            }},
            "indonesia": {"tokens": ["indonesia", "indonesian", "jakarta"], "country_code": "ID", "children": {
                "malacca_strait": {"tokens": ["malacca strait", "strait of malacca"], "level": "strait"},
                "lombok_strait": {"tokens": ["lombok strait"], "level": "strait"},
                "natuna_islands": {"tokens": ["natuna"], "level": "island"},
            }},
            "malaysia": {"tokens": ["malaysia", "malaysian", "kuala lumpur"], "country_code": "MY"},
            "singapore": {"tokens": ["singapore", "singaporean"], "country_code": "SG"},
            "thailand": {"tokens": ["thailand", "thai", "bangkok"], "country_code": "TH"},
            "myanmar": {"tokens": ["myanmar", "burma", "burmese"], "country_code": "MM"},
            "cambodia": {"tokens": ["cambodia", "cambodian", "phnom penh"], "country_code": "KH"},
            "laos": {"tokens": ["laos", "lao", "vientiane"], "country_code": "LA"},
            "brunei": {"tokens": ["brunei"], "country_code": "BN"},
            "timor_leste": {"tokens": ["timor-leste", "east timor"], "country_code": "TL"},
        },
    },
    "south_asia": {
        "tokens": ["south asia"],
        "macro_region": "South Asia",
        "children": {
            "india": {"tokens": ["india", "indian", "new delhi", "modi"], "country_code": "IN", "children": {
                "lac": {"tokens": ["line of actual control", "lac", "ladakh", "galwan"], "level": "border_zone"},
                "kashmir": {"tokens": ["kashmir", "jammu"], "level": "admin_region"},
                "andaman_nicobar": {"tokens": ["andaman", "nicobar"], "level": "island"},
            }},
            "pakistan": {"tokens": ["pakistan", "pakistani", "islamabad"], "country_code": "PK"},
            "bangladesh": {"tokens": ["bangladesh", "bangladeshi", "dhaka"], "country_code": "BD"},
            "sri_lanka": {"tokens": ["sri lanka", "colombo", "hambantota"], "country_code": "LK", "children": {
                "hambantota_port": {"tokens": ["hambantota port"], "level": "port"},
            }},
            "nepal": {"tokens": ["nepal", "kathmandu"], "country_code": "NP"},
            "maldives": {"tokens": ["maldives"], "country_code": "MV"},
        },
    },
    "central_asia": {
        "tokens": ["central asia"],
        "macro_region": "Central Asia",
        "children": {
            "kazakhstan": {"tokens": ["kazakhstan", "kazakh", "astana"], "country_code": "KZ"},
            "uzbekistan": {"tokens": ["uzbekistan", "uzbek", "tashkent"], "country_code": "UZ"},
            "turkmenistan": {"tokens": ["turkmenistan", "turkmen"], "country_code": "TM"},
            "kyrgyzstan": {"tokens": ["kyrgyzstan", "kyrgyz", "bishkek"], "country_code": "KG"},
            "tajikistan": {"tokens": ["tajikistan", "tajik", "dushanbe"], "country_code": "TJ"},
        },
    },
    "middle_east": {
        "tokens": ["middle east", "mideast"],
        "macro_region": "Middle East",
        "children": {
            "iran": {"tokens": ["iran", "iranian", "tehran", "persia"], "country_code": "IR", "children": {
                "strait_of_hormuz": {"tokens": ["strait of hormuz", "hormuz"], "level": "strait"},
            }},
            "israel": {"tokens": ["israel", "israeli", "jerusalem", "tel aviv"], "country_code": "IL", "children": {
                "gaza": {"tokens": ["gaza", "gaza strip"], "level": "admin_region"},
                "west_bank": {"tokens": ["west bank", "judea", "samaria"], "level": "admin_region"},
                "golan_heights": {"tokens": ["golan", "golan heights"], "level": "admin_region"},
            }},
            "saudi_arabia": {"tokens": ["saudi arabia", "saudi", "riyadh"], "country_code": "SA"},
            "uae": {"tokens": ["uae", "emirates", "abu dhabi", "dubai"], "country_code": "AE"},
            "qatar": {"tokens": ["qatar", "doha"], "country_code": "QA"},
            "iraq": {"tokens": ["iraq", "iraqi", "baghdad"], "country_code": "IQ"},
            "syria": {"tokens": ["syria", "syrian", "damascus"], "country_code": "SY"},
            "turkey": {"tokens": ["turkey", "turkish", "ankara", "istanbul", "turkiye"], "country_code": "TR"},
            "lebanon": {"tokens": ["lebanon", "lebanese", "beirut", "hezbollah"], "country_code": "LB"},
            "yemen": {"tokens": ["yemen", "yemeni", "houthi", "sanaa"], "country_code": "YE", "children": {
                "bab_el_mandeb": {"tokens": ["bab el-mandeb", "bab al-mandab"], "level": "strait"},
            }},
            "jordan": {"tokens": ["jordan", "jordanian", "amman"], "country_code": "JO"},
            "kuwait": {"tokens": ["kuwait"], "country_code": "KW"},
            "bahrain": {"tokens": ["bahrain"], "country_code": "BH"},
            "oman": {"tokens": ["oman", "muscat"], "country_code": "OM"},
        },
    },
    "europe": {
        "tokens": ["europe", "european"],
        "macro_region": "Europe",
        "children": {
            "russia": {"tokens": ["russia", "russian", "moscow", "kremlin"], "country_code": "RU", "children": {
                "crimea": {"tokens": ["crimea", "crimean", "sevastopol"], "level": "admin_region"},
                "kaliningrad": {"tokens": ["kaliningrad"], "level": "admin_region"},
                "arctic_russia": {"tokens": ["arctic", "northern sea route"], "level": "region"},
            }},
            "ukraine": {"tokens": ["ukraine", "ukrainian", "kyiv", "kiev"], "country_code": "UA", "children": {
                "donbas": {"tokens": ["donbas", "donetsk", "luhansk", "donbass"], "level": "admin_region"},
                "zaporizhzhia": {"tokens": ["zaporizhzhia", "zaporozhye"], "level": "admin_region"},
            }},
            "uk": {"tokens": ["united kingdom", "britain", "british", "london", "uk"], "country_code": "GB"},
            "france": {"tokens": ["france", "french", "paris", "macron"], "country_code": "FR"},
            "germany": {"tokens": ["germany", "german", "berlin"], "country_code": "DE"},
            "poland": {"tokens": ["poland", "polish", "warsaw"], "country_code": "PL"},
            "baltic_states": {"tokens": ["baltic", "estonia", "latvia", "lithuania"], "level": "region"},
            "finland": {"tokens": ["finland", "finnish", "helsinki"], "country_code": "FI"},
            "sweden": {"tokens": ["sweden", "swedish", "stockholm"], "country_code": "SE"},
            "norway": {"tokens": ["norway", "norwegian", "oslo"], "country_code": "NO"},
            "italy": {"tokens": ["italy", "italian", "rome"], "country_code": "IT"},
            "spain": {"tokens": ["spain", "spanish", "madrid"], "country_code": "ES"},
            "greece": {"tokens": ["greece", "greek", "athens"], "country_code": "GR"},
            "serbia": {"tokens": ["serbia", "serbian", "belgrade"], "country_code": "RS"},
            "kosovo": {"tokens": ["kosovo", "pristina"], "country_code": "XK"},
            "romania": {"tokens": ["romania", "romanian", "bucharest"], "country_code": "RO"},
            "hungary": {"tokens": ["hungary", "hungarian", "budapest"], "country_code": "HU"},
            "netherlands": {"tokens": ["netherlands", "dutch", "amsterdam", "the hague"], "country_code": "NL"},
            "belgium": {"tokens": ["belgium", "belgian", "brussels"], "country_code": "BE"},
        },
    },
    "north_america": {
        "tokens": ["north america"],
        "macro_region": "North America",
        "children": {
            "us": {"tokens": ["united states", "us", "usa", "america", "american", "washington"], "country_code": "US", "children": {
                "guam": {"tokens": ["guam"], "level": "island"},
                "diego_garcia": {"tokens": ["diego garcia"], "level": "military_base"},
                "hawaii": {"tokens": ["hawaii", "pearl harbor"], "level": "admin_region"},
            }},
            "canada": {"tokens": ["canada", "canadian", "ottawa"], "country_code": "CA"},
            "mexico": {"tokens": ["mexico", "mexican", "mexico city"], "country_code": "MX"},
        },
    },
    "south_america": {
        "tokens": ["south america", "latin america"],
        "macro_region": "South America",
        "children": {
            "brazil": {"tokens": ["brazil", "brazilian", "brasilia"], "country_code": "BR"},
            "argentina": {"tokens": ["argentina", "argentine", "buenos aires"], "country_code": "AR"},
            "colombia": {"tokens": ["colombia", "colombian", "bogota"], "country_code": "CO"},
            "venezuela": {"tokens": ["venezuela", "venezuelan", "caracas"], "country_code": "VE"},
            "chile": {"tokens": ["chile", "chilean", "santiago"], "country_code": "CL"},
            "peru": {"tokens": ["peru", "peruvian", "lima"], "country_code": "PE"},
            "cuba": {"tokens": ["cuba", "cuban", "havana"], "country_code": "CU"},
        },
    },
    "africa": {
        "tokens": ["africa", "african"],
        "macro_region": "Africa",
        "children": {
            "egypt": {"tokens": ["egypt", "egyptian", "cairo", "suez"], "country_code": "EG", "children": {
                "suez_canal": {"tokens": ["suez canal"], "level": "strait"},
            }},
            "south_africa": {"tokens": ["south africa", "pretoria", "johannesburg"], "country_code": "ZA"},
            "nigeria": {"tokens": ["nigeria", "nigerian", "abuja", "lagos"], "country_code": "NG"},
            "ethiopia": {"tokens": ["ethiopia", "ethiopian", "addis ababa"], "country_code": "ET"},
            "kenya": {"tokens": ["kenya", "kenyan", "nairobi"], "country_code": "KE"},
            "djibouti": {"tokens": ["djibouti"], "country_code": "DJ"},
            "somalia": {"tokens": ["somalia", "somali", "mogadishu"], "country_code": "SO", "children": {
                "horn_of_africa": {"tokens": ["horn of africa"], "level": "region"},
            }},
            "sudan": {"tokens": ["sudan", "sudanese", "khartoum"], "country_code": "SD"},
            "libya": {"tokens": ["libya", "libyan", "tripoli"], "country_code": "LY"},
            "dr_congo": {"tokens": ["congo", "kinshasa", "drc"], "country_code": "CD"},
            "niger": {"tokens": ["niger", "niamey"], "country_code": "NE"},
            "mali": {"tokens": ["mali", "bamako"], "country_code": "ML"},
            "sahel": {"tokens": ["sahel"], "level": "region"},
        },
    },
    "oceania": {
        "tokens": ["oceania", "pacific islands"],
        "macro_region": "Oceania",
        "children": {
            "australia": {"tokens": ["australia", "australian", "canberra"], "country_code": "AU", "children": {
                "pine_gap": {"tokens": ["pine gap"], "level": "military_base"},
            }},
            "new_zealand": {"tokens": ["new zealand", "wellington"], "country_code": "NZ"},
            "fiji": {"tokens": ["fiji", "suva"], "country_code": "FJ"},
            "png": {"tokens": ["papua new guinea", "png", "port moresby"], "country_code": "PG"},
            "solomon_islands": {"tokens": ["solomon islands"], "country_code": "SB"},
            "tonga": {"tokens": ["tonga"], "country_code": "TO"},
        },
    },
    "indo_pacific": {
        "tokens": ["indo-pacific", "indo pacific"],
        "macro_region": "Indo-Pacific",
        "children": {
            "indian_ocean": {"tokens": ["indian ocean"], "level": "body_of_water"},
            "pacific_ocean": {"tokens": ["pacific ocean", "pacific"], "level": "body_of_water"},
        },
    },
    "arctic": {
        "tokens": ["arctic", "north pole"],
        "macro_region": "Arctic",
        "children": {},
    },
    "global": {
        "tokens": ["global", "worldwide"],
        "macro_region": "Global",
        "children": {},
    },
}


# ---------------------------------------------------------------------------
# Actor hierarchy: organisation_type → key actors
# ---------------------------------------------------------------------------

ACTOR_TREE = {
    "government": {
        "tokens": ["government", "state", "regime", "administration"],
        "children": {},
    },
    "military_org": {
        "tokens": ["military", "armed forces", "army", "navy", "air force"],
        "children": {
            "pla": {"tokens": ["pla", "peoples liberation army"], "country_code": "CN"},
            "pla_navy": {"tokens": ["plan", "pla navy", "chinese navy"], "country_code": "CN"},
            "ccg": {"tokens": ["china coast guard", "ccg"], "country_code": "CN"},
            "us_military": {"tokens": ["us military", "pentagon", "us forces"], "country_code": "US"},
            "us_navy": {"tokens": ["us navy", "seventh fleet", "pacific fleet"], "country_code": "US"},
            "jsdf": {"tokens": ["jsdf", "japan self-defense", "jmsdf"], "country_code": "JP"},
            "russian_military": {"tokens": ["russian military", "russian armed forces"], "country_code": "RU"},
            "nato_forces": {"tokens": ["nato forces", "nato allied"], "country_code": ""},
        },
    },
    "intelligence_org": {
        "tokens": ["intelligence agency", "intelligence service"],
        "children": {
            "cia": {"tokens": ["cia", "central intelligence agency"], "country_code": "US"},
            "fbi": {"tokens": ["fbi", "federal bureau"], "country_code": "US"},
            "nsa": {"tokens": ["nsa", "national security agency"], "country_code": "US"},
            "mss": {"tokens": ["mss", "ministry of state security"], "country_code": "CN"},
            "fsb": {"tokens": ["fsb", "federal security service"], "country_code": "RU"},
            "gru": {"tokens": ["gru", "military intelligence"], "country_code": "RU"},
            "mi6": {"tokens": ["mi6", "sis", "secret intelligence service"], "country_code": "GB"},
            "mossad": {"tokens": ["mossad"], "country_code": "IL"},
            "isi": {"tokens": ["isi", "inter-services intelligence"], "country_code": "PK"},
        },
    },
    "intl_org": {
        "tokens": ["international organization"],
        "children": {
            "un_org": {"tokens": ["united nations", "un"], "children": {
                "unsc": {"tokens": ["security council", "unsc"]},
                "unga": {"tokens": ["general assembly", "unga"]},
                "iaea": {"tokens": ["iaea", "international atomic energy"]},
            }},
            "nato": {"tokens": ["nato", "north atlantic treaty"]},
            "eu_org": {"tokens": ["european union", "eu", "european commission"]},
            "asean_org": {"tokens": ["asean", "association of southeast asian"]},
            "brics": {"tokens": ["brics"]},
            "sco": {"tokens": ["sco", "shanghai cooperation"]},
            "aukus_org": {"tokens": ["aukus"]},
            "quad_org": {"tokens": ["quad", "quadrilateral"]},
            "imf_org": {"tokens": ["imf", "international monetary fund"]},
            "world_bank_org": {"tokens": ["world bank"]},
        },
    },
    "non_state_actor": {
        "tokens": ["non-state", "non state", "militant", "insurgent"],
        "children": {
            "hamas": {"tokens": ["hamas"], "country_code": "PS"},
            "hezbollah_actor": {"tokens": ["hezbollah"], "country_code": "LB"},
            "houthi": {"tokens": ["houthi", "ansar allah"], "country_code": "YE"},
            "wagner": {"tokens": ["wagner", "wagner group", "pmc"], "country_code": "RU"},
            "isis": {"tokens": ["isis", "isil", "islamic state", "daesh"]},
            "taliban": {"tokens": ["taliban"], "country_code": "AF"},
        },
    },
}


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def _slug(name):
    """Convert a key to a clean anchor name."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _build_dimefil(tree, schema, parent_key=None, depth=0):
    """Recursively walk the DIMEFIL tree and emit anchors."""
    level_map = {0: "L1", 1: "L2", 2: "L3"}

    for key, node in tree.items():
        anchor_name = _slug(key)
        entry = {
            "type": "relation",
            "tokens": node.get("tokens", [key]),
            "level": level_map.get(depth, f"L{depth}"),
        }
        if parent_key:
            entry["parent"] = parent_key

        schema[anchor_name] = entry

        children = node.get("children", {})
        if children:
            _build_dimefil(children, schema, parent_key=anchor_name, depth=depth + 1)


def _build_locations(tree, schema, parent_key=None, depth=0):
    """Recursively walk the location tree and emit anchors."""
    level_map = {0: "region", 1: "country", 2: "sub_feature"}

    for key, node in tree.items():
        anchor_name = _slug(key)
        default_level = level_map.get(depth, "other")
        entry = {
            "type": "location",
            "tokens": node.get("tokens", [key]),
            "level": node.get("level", default_level),
        }
        if parent_key:
            entry["parent"] = parent_key
        if node.get("country_code"):
            entry["country_code"] = node["country_code"]
        if node.get("macro_region"):
            entry["macro_region"] = node["macro_region"]
        elif parent_key and parent_key in schema:
            mr = schema[parent_key].get("macro_region")
            if mr:
                entry["macro_region"] = mr

        schema[anchor_name] = entry

        children = node.get("children", {})
        if children:
            _build_locations(children, schema, parent_key=anchor_name, depth=depth + 1)


def _build_actors(tree, schema, parent_key=None, depth=0):
    """Recursively walk the actor tree and emit anchors."""
    for key, node in tree.items():
        anchor_name = _slug(key)
        entry = {
            "type": "actor",
            "tokens": node.get("tokens", [key]),
        }
        if parent_key:
            entry["parent"] = parent_key
        if node.get("country_code"):
            entry["country_code"] = node["country_code"]

        schema[anchor_name] = entry

        children = node.get("children", {})
        if children:
            _build_actors(children, schema, parent_key=anchor_name, depth=depth + 1)


def generate_schema():
    """Generate the complete DIMEFIL + geo + actor schema."""
    schema = {}

    _build_dimefil(DIMEFIL_TREE, schema)
    _build_locations(LOCATION_TREE, schema)
    _build_actors(ACTOR_TREE, schema)

    return schema


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate DIMEFIL cognition schema")
    parser.add_argument("--output", "-o",
                        default=str(Path(__file__).parent.parent / "data" / "dimefil_schema.json"),
                        help="Output path for the schema JSON")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    schema = generate_schema()

    with open(args.output, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {len(schema)} anchors to {args.output}")

    if args.stats:
        from collections import Counter
        types = Counter(v["type"] for v in schema.values())
        parents = sum(1 for v in schema.values() if "parent" in v)
        depths = Counter()
        for name, info in schema.items():
            d = 0
            cur = name
            while schema.get(cur, {}).get("parent"):
                cur = schema[cur]["parent"]
                d += 1
            depths[d] += 1

        print(f"\nBy type:  {dict(types)}")
        print(f"With parent: {parents}")
        print(f"Depth distribution: {dict(sorted(depths.items()))}")

        total_tokens = sum(len(v.get("tokens", [])) for v in schema.values())
        print(f"Total token entries: {total_tokens}")


if __name__ == "__main__":
    main()
