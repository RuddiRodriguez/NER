import random, pandas as pd, json
import caas_jupyter_tools

random.seed(7)

langs = ["en","es","fr",""]  # blank sometimes

hotels = {
 "en": ["Maple Harbor Hotel","The Marigold Grand","Cedar & Stone Suites","Riverview Boutique Inn","Skyline Court Hotel",
        "Aurora Garden Lodge","Seabreeze Manor","Golden Key Hotel","Northbridge Plaza","Willow Creek Stay"],
 "es": ["Hotel Bahía Azul","Gran Hotel Marigold","Suites Cedro y Piedra","Posada Mirador del Río","Hotel Patio Skyline",
        "Lodge Jardín Aurora","Hostal Brisa Marina","Hotel Llave Dorada","Plaza Puente Norte","Estancia Arroyo Sauce"],
 "fr": ["Hôtel Port-Érable","Le Grand Marigold","Suites Cèdre et Pierre","Auberge Vue-sur-Rivière","Hôtel Cour Skyline",
        "Lodge Jardin Aurore","Manoir Brise-Marine","Hôtel Clé d’Or","Plaza Pont-du-Nord","Séjour Ruisseau-Saule"]
}
pois = {
 "en": ["Old Town Market Hall","Riverside Museum of Art","Central Station","Liberty Square","Harborfront Promenade",
        "Greenleaf Park","Sunset Lighthouse","City Aquarium","Royal Botanical Garden","Stonebridge Cathedral"],
 "es": ["Mercado del Casco Antiguo","Museo de Arte del Río","Estación Central","Plaza Libertad","Paseo del Puerto",
        "Parque Hoja Verde","Faro del Atardecer","Acuario de la Ciudad","Jardín Botánico Real","Catedral de Puente de Piedra"],
 "fr": ["Halles du Vieux Quartier","Musée d’Art de la Rivière","Gare Centrale","Place de la Liberté","Promenade du Port",
        "Parc Feuille-Verte","Phare du Couchant","Aquarium de la Ville","Jardin botanique royal","Cathédrale du Pont-de-Pierre"]
}
addresses = {
 "en": ["204 Kingsway Avenue, Edgemont","12 Harbor Street, Seaview","8-10 Wrenford Road, Westbridge","77 Willow Lane, Northbridge",
        "5 Liberty Square, Old Town","19 Riverside Drive, Brookhaven","301 Skyline Way, Crestfield","42 Greenleaf Park Rd, Hillgate",
        "10 Stonebridge St, Old Town","88 Oceanview Blvd, Harbor City"],
 "es": ["Avenida Kingsway 204, Edgemont","Calle Puerto 12, Seaview","Carretera Wrenford 8-10, Westbridge","Calle Sauce 77, Northbridge",
        "Plaza Libertad 5, Casco Antiguo","Paseo del Río 19, Brookhaven","Camino Skyline 301, Crestfield","Av. Parque Hoja Verde 42, Hillgate",
        "Calle Puente de Piedra 10, Casco Antiguo","Bulevar Vista al Mar 88, Harbor City"],
 "fr": ["204 avenue Kingsway, Edgemont","12 rue du Port, Seaview","8-10 route de Wrenford, Westbridge","77 allée des Saules, Northbridge",
        "5 place de la Liberté, Vieux Quartier","19 promenade de la Rivière, Brookhaven","301 voie Skyline, Crestfield","42 route du Parc Feuille-Verte, Hillgate",
        "10 rue du Pont-de-Pierre, Vieux Quartier","88 boulevard Vue-sur-Mer, Harbor City"]
}

def make_desc(lang, hotel=None, addr=None, poi=None):
    if lang=="en":
        parts=[]
        parts.append(f"Welcome to {hotel}, a cozy stay with fast Wi‑Fi and a quiet lobby." if hotel else
                     "A cozy stay with fast Wi‑Fi and a quiet lobby.")
        if addr: parts.append(f"Find us at {addr}.")
        if poi: parts.append(f"Just minutes from {poi} and great local cafés.")
        return " ".join(parts)
    if lang=="es":
        parts=[]
        parts.append(f"Bienvenido a {hotel}, una estancia cómoda con Wi‑Fi rápido y recepción 24h." if hotel else
                     "Una estancia cómoda con Wi‑Fi rápido y recepción 24h.")
        if addr: parts.append(f"Nos encuentras en {addr}.")
        if poi: parts.append(f"A pocos minutos de {poi} y de cafeterías locales.")
        return " ".join(parts)
    if lang=="fr":
        parts=[]
        parts.append(f"Bienvenue à {hotel}, un séjour confortable avec Wi‑Fi rapide et réception 24h/24." if hotel else
                     "Un séjour confortable avec Wi‑Fi rapide et réception 24h/24.")
        if addr: parts.append(f"Adresse : {addr}.")
        if poi: parts.append(f"À quelques minutes de {poi} et de cafés locaux.")
        return " ".join(parts)
    parts=[]
    parts.append(f"Stay at {hotel} with Wi‑Fi and friendly staff." if hotel else
                 "Comfortable stay with Wi‑Fi and friendly staff.")
    if addr: parts.append(f"Address: {addr}.")
    if poi: parts.append(f"Near {poi}.")
    return " ".join(parts)

rows=[]
for i in range(50):
    lang = random.choice(langs)
    content_lang = lang if lang in ["en","es","fr"] else random.choice(["en","es","fr"])

    r = random.random()
    hotel = addr = poi = None

    if r < 0.45:
        hotel = random.choice(hotels[content_lang])
        addr = random.choice(addresses[content_lang])
        poi = random.choice(pois[content_lang])
    elif r < 0.70:
        pick = random.choice(["hotel_poi","hotel_addr","addr_poi"])
        if pick=="hotel_poi":
            hotel = random.choice(hotels[content_lang])
            poi = random.choice(pois[content_lang])
        elif pick=="hotel_addr":
            hotel = random.choice(hotels[content_lang])
            addr = random.choice(addresses[content_lang])
        else:
            addr = random.choice(addresses[content_lang])
            poi = random.choice(pois[content_lang])
    elif r < 0.85:
        pick = random.choice(["hotel","addr","poi"])
        if pick=="hotel":
            hotel = random.choice(hotels[content_lang])
        elif pick=="addr":
            addr = random.choice(addresses[content_lang])
        else:
            poi = random.choice(pois[content_lang])

    desc = make_desc(content_lang, hotel=hotel, addr=addr, poi=poi)

    rows.append({
        "id": f"ex_{i+1:03d}",
        "language": lang,  # sometimes blank
        "hotel_name": hotel or "",
        "address": addr or "",
        "landmark": poi or "",
        "description": desc
    })

df = pd.DataFrame(rows, columns=["id","language","hotel_name","address","landmark","description"])

out_csv = "data/toy_hotel_ner_50.csv"
df.to_csv(out_csv, index=False)

out_jsonl = "data/toy_hotel_ner_50.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as f:
    for rec in rows:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

caas_jupyter_tools.display_dataframe_to_user("Toy multilingual hotel NER dataset (50 rows)", df)

(out_csv, out_jsonl)

