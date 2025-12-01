

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from gtts import gTTS
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Medicinal Plant Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with beautiful green theme
st.markdown("""
    <style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);
    }

    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        color: #1B5E20;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.15);
        animation: fadeIn 1s ease-in;
    }

    .sub-header {
        font-size: 1.3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Beautiful prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(67, 160, 71, 0.4);
        animation: scaleIn 0.5s ease-out;
    }

    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    .prediction-name {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .confidence-score {
        font-size: 1.8rem;
        opacity: 0.95;
        font-weight: 500;
    }

    /* Info cards */
    .info-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .info-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .info-title {
        font-size: 1.6rem;
        color: #2E7D32;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #66BB6A;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }

    .info-content {
        font-size: 1.1rem;
        color: #424242;
        line-height: 1.8;
        margin-bottom: 0.5rem;
    }

    .scientific-name {
        font-style: italic;
        font-size: 1.3rem;
        color: #558B2F;
        margin: 1rem 0;
        padding: 0.5rem;
        background: #F1F8E9;
        border-left: 4px solid #66BB6A;
        border-radius: 5px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #C8E6C9 0%, #A5D6A7 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #1B5E20;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(67, 160, 71, 0.3);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #388E3C 0%, #4CAF50 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(67, 160, 71, 0.4);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Success/Warning/Error boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 5px;
    }

    /* Image container */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #66BB6A, transparent);
    }

    /* Tips box */
    .tips-box {
        background: linear-gradient(135deg, #81C784 0%, #A5D6A7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .tips-box h3 {
        color: white;
        margin-bottom: 1rem;
    }

    .tips-box ul {
        list-style-type: none;
        padding-left: 0;
    }

    .tips-box li {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }

    .tips-box li:before {
        content: "✓";
        position: absolute;
        left: 0;
        font-weight: bold;
        color: #1B5E20;
    }

    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: #43A047;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Extended plant information database
PLANT_INFO = {
    'Arive-Dantu': {
        'scientific_name': 'Amaranthus viridis',
        'common_names': 'Green Amaranth, Slender Amaranth, Pigweed',
        'family': 'Amaranthaceae',
        'uses': 'Rich in iron and calcium, used for treating anemia, anti-inflammatory properties, helps in wound healing, blood purifier',
        'description': 'A highly nutritious leafy vegetable cultivated throughout India. The leaves are rich in vitamins A, C, and K, along with essential minerals.',
        'parts_used': 'Leaves, stems, seeds',
        'preparation': 'Consumed as cooked vegetable, juice extract, or powder form',
        'precautions': 'Generally safe, but people with kidney problems should consume in moderation due to oxalate content'
    },
    'Basale': {
        'scientific_name': 'Basella alba',
        'common_names': 'Malabar Spinach, Indian Spinach, Ceylon Spinach',
        'family': 'Basellaceae',
        'uses': 'Natural laxative, cooling effect on body, improves digestion, beneficial for skin health, helps in treating ulcers',
        'description': 'A fast-growing climbing vine with succulent leaves, commonly used as a vegetable in tropical regions.',
        'parts_used': 'Leaves, tender stems',
        'preparation': 'Cooked as vegetable, leaf paste for external application',
        'precautions': 'Safe for regular consumption, pregnant women should consult doctor'
    },
    'Betel': {
        'scientific_name': 'Piper betle',
        'common_names': 'Betel Leaf, Paan, Tambul',
        'family': 'Piperaceae',
        'uses': 'Digestive aid, antiseptic properties, breath freshener, wound healing, relieves headache, stimulates appetite',
        'description': 'A sacred plant used in religious ceremonies and traditional medicine across South Asia for thousands of years.',
        'parts_used': 'Fresh leaves',
        'preparation': 'Chewed fresh, leaf extract, essential oil',
        'precautions': 'Excessive consumption may cause mouth irritation, avoid with tobacco'
    },
    'Crape_Jasmine': {
        'scientific_name': 'Tabernaemontana divaricata',
        'common_names': 'Crepe Jasmine, Pinwheel Flower, East Indian Rosebay',
        'family': 'Apocynaceae',
        'uses': 'Treats eye disorders, heals wounds, manages skin diseases, anti-inflammatory, antimicrobial properties',
        'description': 'An ornamental evergreen shrub with fragrant white flowers, widely used in traditional medicine systems.',
        'parts_used': 'Leaves, flowers, roots, latex',
        'preparation': 'Leaf paste, decoction, latex application',
        'precautions': 'Use under expert guidance, latex may cause skin irritation'
    },
    'Curry': {
        'scientific_name': 'Murraya koenigii',
        'common_names': 'Curry Leaf, Sweet Neem, Kadi Patta',
        'family': 'Rutaceae',
        'uses': 'Controls diabetes, improves digestion, promotes hair growth, antioxidant, manages cholesterol, prevents premature greying',
        'description': 'An essential aromatic herb in Indian cuisine, also valued for its numerous medicinal properties.',
        'parts_used': 'Fresh leaves, bark, roots',
        'preparation': 'Consumed fresh in cooking, leaf powder, hair oil',
        'precautions': 'Generally safe, diabetics should monitor blood sugar levels'
    },
    'Drumstick': {
        'scientific_name': 'Moringa oleifera',
        'common_names': 'Moringa, Horseradish Tree, Drumstick Tree',
        'family': 'Moringaceae',
        'uses': 'Nutrient-rich superfood, anti-diabetic, anti-inflammatory, improves bone health, boosts immunity, purifies blood',
        'description': 'Known as the "miracle tree", every part of moringa is edible and packed with nutrients exceeding many common foods.',
        'parts_used': 'Leaves, pods, flowers, seeds, roots',
        'preparation': 'Cooked vegetable, leaf powder, seed oil, tea',
        'precautions': 'Safe in normal amounts, avoid excessive root consumption during pregnancy'
    },
    'Fenugreek': {
        'scientific_name': 'Trigonella foenum-graecum',
        'common_names': 'Methi, Greek Hay',
        'family': 'Fabaceae',
        'uses': 'Manages diabetes, increases breast milk production, improves digestive health, controls cholesterol, promotes hair growth',
        'description': 'A highly valued medicinal herb and spice used in Ayurvedic and Chinese medicine for centuries.',
        'parts_used': 'Seeds, leaves',
        'preparation': 'Soaked seeds, powder, leaf vegetable, tea',
        'precautions': 'May interact with diabetes medications, avoid in large amounts during pregnancy'
    },
    'Guava': {
        'scientific_name': 'Psidium guajava',
        'common_names': 'Guava, Amrood, Common Guava',
        'family': 'Myrtaceae',
        'uses': 'Treats diarrhea, boosts immunity, controls diabetes, improves heart health, aids weight loss, promotes healthy skin',
        'description': 'A tropical fruit tree whose leaves contain more vitamin C than the fruit itself, widely used in herbal medicine.',
        'parts_used': 'Leaves, fruits, bark',
        'preparation': 'Fresh fruit, leaf tea, decoction',
        'precautions': 'Generally safe, excessive consumption may cause constipation'
    },
    'Hibiscus': {
        'scientific_name': 'Hibiscus rosa-sinensis',
        'common_names': 'Chinese Hibiscus, Shoe Flower, Gudhal',
        'family': 'Malvaceae',
        'uses': 'Promotes hair growth, controls blood pressure, improves liver health, treats menstrual problems, natural coolant',
        'description': 'A beautiful flowering plant with significant medicinal value, especially renowned for hair care in traditional medicine.',
        'parts_used': 'Flowers, leaves',
        'preparation': 'Flower paste, hair oil, tea, powder',
        'precautions': 'May lower blood pressure, pregnant women should avoid'
    },
    'Indian_Beech': {
        'scientific_name': 'Pongamia pinnata',
        'common_names': 'Karanja, Pongam, Indian Beech',
        'family': 'Fabaceae',
        'uses': 'Treats skin diseases, heals wounds, antiseptic properties, anti-inflammatory, manages diabetes, relieves joint pain',
        'description': 'A medium-sized tree with seeds and leaves that have powerful medicinal properties, especially for skin ailments.',
        'parts_used': 'Seeds, leaves, bark, oil',
        'preparation': 'Seed oil, leaf paste, decoction',
        'precautions': 'External use preferred, internal use under expert guidance only'
    },
    'Indian_Mustard': {
        'scientific_name': 'Brassica juncea',
        'common_names': 'Brown Mustard, Rai, Sarson',
        'family': 'Brassicaceae',
        'uses': 'Relieves respiratory problems, provides pain relief, improves blood circulation, treats skin conditions, aids digestion',
        'description': 'Cultivated for its pungent seeds and nutritious leaves, widely used as a spice and medicinal plant.',
        'parts_used': 'Seeds, leaves, oil',
        'preparation': 'Seed powder, mustard oil, cooked greens, paste',
        'precautions': 'May cause skin irritation in sensitive individuals'
    },
    'Jackfruit': {
        'scientific_name': 'Artocarpus heterophyllus',
        'common_names': 'Jack Tree, Kathal, Panasa',
        'family': 'Moraceae',
        'uses': 'Boosts immunity, improves digestive health, provides sustained energy, strengthens bones, controls blood pressure',
        'description': 'The largest tree-borne fruit in the world, with leaves that have medicinal properties for various ailments.',
        'parts_used': 'Fruit, seeds, leaves, bark',
        'preparation': 'Fresh fruit, cooked vegetable, leaf decoction',
        'precautions': 'Generally safe, diabetics should monitor blood sugar after consumption'
    },
    'Jamaica_Cherry-Gasagase': {
        'scientific_name': 'Muntingia calabura',
        'common_names': 'Jamaican Cherry, Singapore Cherry, Gasagase',
        'family': 'Muntingiaceae',
        'uses': 'Anti-inflammatory, rich in antioxidants, provides pain relief, treats headaches, reduces fever',
        'description': 'A fast-growing ornamental tree with small sweet fruits and leaves with significant medicinal properties.',
        'parts_used': 'Fruits, leaves, flowers',
        'preparation': 'Fresh fruit, leaf tea, decoction',
        'precautions': 'Safe for general consumption'
    },
    'Jamun': {
        'scientific_name': 'Syzygium cumini',
        'common_names': 'Java Plum, Black Plum, Jamun',
        'family': 'Myrtaceae',
        'uses': 'Excellent for diabetes control, improves digestive health, antibacterial properties, strengthens teeth and gums, purifies blood',
        'description': 'A traditional remedy for diabetes in Ayurvedic medicine, the seeds are particularly valued for blood sugar management.',
        'parts_used': 'Fruits, seeds, leaves, bark',
        'preparation': 'Fresh fruit, seed powder, leaf tea, bark decoction',
        'precautions': 'May lower blood sugar significantly, monitor levels'
    },
    'Jasmine': {
        'scientific_name': 'Jasminum officinale',
        'common_names': 'Common Jasmine, Poets Jasmine, Chameli',
        'family': 'Oleaceae',
        'uses': 'Promotes relaxation, improves skin health, used in aromatherapy, antioxidant properties, relieves stress and anxiety',
        'description': 'A fragrant flowering plant used in perfumery and traditional medicine for its calming and therapeutic properties.',
        'parts_used': 'Flowers, leaves, essential oil',
        'preparation': 'Flower tea, essential oil, aromatherapy',
        'precautions': 'Generally safe, some may have allergic reactions to fragrance'
    },
    'Karanda': {
        'scientific_name': 'Carissa carandas',
        'common_names': 'Bengal Currant, Christ Thorn, Karonda',
        'family': 'Apocynaceae',
        'uses': 'Treats digestive problems, rich in antioxidants, antimicrobial properties, improves appetite, beneficial for anemia',
        'description': 'A berry-producing shrub with tangy fruits rich in vitamin C and iron, used in traditional medicine.',
        'parts_used': 'Fruits, leaves, roots',
        'preparation': 'Fresh fruit, pickle, juice, leaf decoction',
        'precautions': 'Generally safe for consumption'
    },
    'Lemon': {
        'scientific_name': 'Citrus limon',
        'common_names': 'Lemon, Nimbu',
        'family': 'Rutaceae',
        'uses': 'Excellent source of vitamin C, boosts immunity, aids digestion, promotes weight loss, improves skin health, detoxifies body',
        'description': 'A citrus fruit tree highly valued for its juice and peel, extensively used in culinary and medicinal applications.',
        'parts_used': 'Fruit, juice, peel, leaves',
        'preparation': 'Fresh juice, tea, essential oil, zest',
        'precautions': 'May erode tooth enamel if consumed excessively'
    },
    'Mango': {
        'scientific_name': 'Mangifera indica',
        'common_names': 'Mango, Aam, King of Fruits',
        'family': 'Anacardiaceae',
        'uses': 'Improves digestive health, boosts immunity, antioxidant properties, promotes eye health, beneficial for diabetes management',
        'description': 'Known as the king of fruits, mango leaves are used in traditional medicine for various therapeutic purposes.',
        'parts_used': 'Leaves, fruit, bark, flowers',
        'preparation': 'Fresh fruit, leaf powder, tea, bark decoction',
        'precautions': 'Unripe fruit in excess may cause stomach upset'
    },
    'Mexican_Mint': {
        'scientific_name': 'Plectranthus amboinicus',
        'common_names': 'Cuban Oregano, Indian Borage, Patta Ajwain',
        'family': 'Lamiaceae',
        'uses': 'Provides cough relief, treats respiratory problems, anti-inflammatory, improves digestion, relieves cold and fever',
        'description': 'An aromatic herb with thick succulent leaves, widely used in traditional medicine for respiratory ailments.',
        'parts_used': 'Fresh leaves',
        'preparation': 'Leaf juice, decoction, chewed fresh',
        'precautions': 'Safe in moderate amounts'
    },
    'Mint': {
        'scientific_name': 'Mentha',
        'common_names': 'Pudina, Peppermint, Spearmint',
        'family': 'Lamiaceae',
        'uses': 'Aids digestion, relieves headaches, promotes respiratory health, freshens breath, reduces nausea, improves concentration',
        'description': 'A popular aromatic herb with cooling properties, extensively used in culinary and medicinal preparations worldwide.',
        'parts_used': 'Fresh leaves, essential oil',
        'preparation': 'Fresh leaves in cooking, tea, juice, essential oil',
        'precautions': 'Generally safe, may cause heartburn in some individuals'
    },
    'Neem': {
        'scientific_name': 'Azadirachta indica',
        'common_names': 'Neem, Margosa, Divine Tree',
        'family': 'Meliaceae',
        'uses': 'Powerful antibacterial, antifungal, blood purifier, treats skin diseases, dental care, controls diabetes, boosts immunity',
        'description': 'Considered sacred in India, neem is called "the village pharmacy" due to its extensive medicinal properties.',
        'parts_used': 'Leaves, bark, seeds, oil, flowers',
        'preparation': 'Leaf paste, juice, oil, powder, tooth brushing twig',
        'precautions': 'Safe externally, internal use should be in moderation'
    },
    'Oleander': {
        'scientific_name': 'Nerium oleander',
        'common_names': 'Kaner, Rose Bay, Adelfa',
        'family': 'Apocynaceae',
        'uses': 'Treats heart conditions (under strict medical supervision), skin diseases, antibacterial properties',
        'description': 'A beautiful ornamental plant with potent medicinal properties, requires expert handling due to toxicity.',
        'parts_used': 'Leaves, flowers (with extreme caution)',
        'preparation': 'Only by qualified practitioners in controlled doses',
        'precautions': 'HIGHLY TOXIC - Use only under expert medical supervision, keep away from children and pets'
    },
    'Parijata': {
        'scientific_name': 'Nyctanthes arbor-tristis',
        'common_names': 'Night Jasmine, Harsingar, Coral Jasmine',
        'family': 'Oleaceae',
        'uses': 'Treats arthritis effectively, reduces fever, relieves sciatica pain, anti-inflammatory, manages anxiety, improves immunity',
        'description': 'A sacred tree with night-blooming fragrant flowers, highly valued in Ayurvedic medicine for joint disorders.',
        'parts_used': 'Leaves, flowers, bark, seeds',
        'preparation': 'Leaf juice, decoction, powder, paste',
        'precautions': 'Generally safe, consult doctor if on other medications'
    },
    'Peepal': {
        'scientific_name': 'Ficus religiosa',
        'common_names': 'Sacred Fig, Bodhi Tree, Ashvattha',
        'family': 'Moraceae',
        'uses': 'Treats respiratory problems, heals skin diseases, promotes wound healing, manages diabetes, improves heart health',
        'description': 'One of the most sacred trees in Hinduism and Buddhism, with extensive use in traditional medicine.',
        'parts_used': 'Leaves, bark, fruits, latex',
        'preparation': 'Leaf powder, bark decoction, fruit consumption',
        'precautions': 'Safe for general use'
    },
    'Pomegranate': {
        'scientific_name': 'Punica granatum',
        'common_names': 'Anar, Pomegranate',
        'family': 'Lythraceae',
        'uses': 'Powerful antioxidant, promotes heart health, anti-cancer properties, improves memory, manages blood pressure, anti-aging',
        'description': 'A fruit-bearing shrub with exceptional health benefits, considered one of the healthiest fruits in the world.',
        'parts_used': 'Fruit, juice, peel, leaves, flowers',
        'preparation': 'Fresh fruit, juice, peel powder, tea',
        'precautions': 'May interact with certain medications, consult doctor'
    },
    'Rasna': {
        'scientific_name': 'Pluchea lanceolata',
        'common_names': 'Rasna, Indian Camphor',
        'family': 'Asteraceae',
        'uses': 'Excellent for arthritis treatment, relieves joint pain, anti-inflammatory, treats respiratory disorders, reduces fever',
        'description': 'An important herb in Ayurvedic medicine, particularly valued for treating rheumatic and joint conditions.',
        'parts_used': 'Leaves, roots',
        'preparation': 'Decoction, powder, paste, medicated oil',
        'precautions': 'Safe under proper guidance'
    },
    'Rose_apple': {
        'scientific_name': 'Syzygium jambos',
        'common_names': 'Rose Apple, Jambu, Malabar Plum',
        'family': 'Myrtaceae',
        'uses': 'Improves digestive health, helps control diabetes, antimicrobial properties, cooling effect, improves brain function',
        'description': 'A tropical fruit tree with rose-scented fruits and leaves that have various medicinal applications.',
        'parts_used': 'Fruits, leaves, bark',
        'preparation': 'Fresh fruit, leaf tea, bark decoction',
        'precautions': 'Generally safe for consumption'
    },
    'Roxburgh_fig': {
        'scientific_name': 'Ficus auriculata',
        'common_names': 'Elephant Ear Fig, Roxburgh Fig',
        'family': 'Moraceae',
        'uses': 'Treats digestive problems effectively, promotes wound healing, improves respiratory health, manages diabetes',
        'description': 'A fig tree with large leaves resembling elephant ears, valued for its medicinal fruits and leaves.',
        'parts_used': 'Fruits, leaves, latex',
        'preparation': 'Ripe fruit, leaf paste, latex application',
        'precautions': 'Safe for general use'
    },
    'Sandalwood': {
        'scientific_name': 'Santalum album',
        'common_names': 'Chandan, White Sandalwood',
        'family': 'Santalaceae',
        'uses': 'Excellent for skin care, provides cooling effect, antiseptic properties, used in aromatherapy, treats acne, anti-aging',
        'description': 'A precious aromatic wood highly valued in traditional medicine, perfumery, and religious ceremonies.',
        'parts_used': 'Heartwood, essential oil',
        'preparation': 'Paste, essential oil, powder, incense',
        'precautions': 'Generally safe for external use'
    },
    'Tulsi': {
        'scientific_name': 'Ocimum tenuiflorum',
        'common_names': 'Holy Basil, Tulsi, Sacred Basil',
        'family': 'Lamiaceae',
        'uses': 'Powerful immunity booster, relieves stress, treats respiratory infections, reduces fever, improves heart health, adaptogenic properties',
        'description': 'Considered the "Queen of Herbs", tulsi is a sacred plant in Hinduism with remarkable medicinal properties.',
        'parts_used': 'Leaves, seeds, whole plant',
        'preparation': 'Fresh leaves, tea, juice, essential oil',
        'precautions': 'Generally very safe, may lower blood sugar levels'
    }
}


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('models/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def is_valid_leaf_image(image):
    """
    Simple and aggressive check for leaf images
    Returns: (is_valid, reason)
    """
    img_array = np.array(image.convert('RGB'))

    # Check 1: Image dimensions
    width, height = image.size
    if width < 50 or height < 50:
        return False, "Image is too small. Please upload a larger image."

    # Check 2: Calculate average colors
    avg_color = img_array.mean(axis=(0, 1))
    red, green, blue = avg_color

    # AGGRESSIVE CHECK: Leaves MUST have dominant green channel
    # OR be brownish (dried leaves)
    is_greenish = (green > red * 1.1 and green > blue * 1.1)  # Green dominant
    is_dried_brown = (red > 100 and green > 80 and abs(red - green) < 30 and blue < green)

    if not (is_greenish or is_dried_brown):
        return False, "NOT A LEAF - No green plant color detected. This app ONLY works with plant leaves!"

    # Check 3: Brightness
    brightness = img_array.mean()
    if brightness < 30:
        return False, "Image is too dark."
    if brightness > 240:
        return False, "Image is too bright."

    return True, "Valid"


def predict_plant(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    return predictions[0]


def create_confidence_chart(predictions, class_names):
    """Create an interactive confidence chart"""
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    top_5_classes = [class_names[i] for i in top_5_idx]
    top_5_scores = [predictions[i] * 100 for i in top_5_idx]

    colors = ['#43A047' if i == 0 else '#81C784' for i in range(5)]

    fig = go.Figure(data=[
        go.Bar(
            x=top_5_scores,
            y=top_5_classes,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#2E7D32', width=2)
            ),
            text=[f'{score:.2f}%' for score in top_5_scores],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black'),
        )
    ])

    fig.update_layout(
        title={
            'text': "Top 5 Predictions",
            'font': {'size': 20, 'color': '#1B5E20', 'family': 'Arial Black'}
        },
        xaxis_title="Confidence (%)",
        yaxis_title="Plant Species",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#2E7D32'),
        xaxis=dict(gridcolor='rgba(46, 125, 50, 0.1)'),
        yaxis=dict(gridcolor='rgba(46, 125, 50, 0.1)')
    )

    return fig


def text_to_speech(text, plant_name):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None


def main():
    # Header with animation
    st.markdown('<p class="main-header">🌿 Medicinal Plant Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Identify medicinal plants by their leaf characteristics using AI-powered deep learning</p>',
        unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📚 About")
        st.info(
            """
            This AI-powered application identifies **30 different medicinal plants** 
            based on their leaf characteristics using advanced deep learning.

            🎯 **Model Accuracy:** 97.59%
            🧠 **AI Model:** Convolutional Neural Network
            🌍 **Database:** 30 Medicinal Plants

            **How to use:**
            1. Upload a clear image of a leaf
            2. Get instant AI-powered identification
            3. Learn about medicinal properties
            4. Listen to plant information
            """
        )

        st.markdown("---")
        st.subheader("🌱 Supported Plants (30)")

        st.markdown(
            """
            <div style='background: white; padding: 1rem; border-radius: 10px; max-height: 300px; overflow-y: auto;'>
            """,
            unsafe_allow_html=True
        )

        for plant in sorted(PLANT_INFO.keys()):
            st.markdown(f"✅ {plant}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔬 Key Features")
        st.markdown(
            """
            <span class="feature-badge">🤖 AI-Powered</span>
            <span class="feature-badge">📊 97.59% Accurate</span>
            <span class="feature-badge">🔊 Text-to-Speech</span>
            
            """,
            unsafe_allow_html=True
        )

    # Load model
    model = load_model()

    if model is None:
        st.error("⚠️ Failed to load the model. Please check if 'models/best_model.h5' exists.")
        return

    # Get class names
    class_names = sorted(PLANT_INFO.keys())

    # File uploader
    st.markdown("### 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a medicinal plant leaf for AI analysis"
    )

    if uploaded_file is not None:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🖼️ Input Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

            # INLINE VALIDATION CHECK - v2.0
            img_arr = np.array(image.convert('RGB'))
            r_avg, g_avg, b_avg = img_arr.mean(axis=(0, 1))

            # Simple green check
            is_leaf_color = (g_avg > r_avg * 1.05) or (
                        r_avg > 100 and g_avg > 80 and abs(r_avg - g_avg) < 35 and b_avg < g_avg)

            if not is_leaf_color:
                st.error(
                    f"⚠️ **Image Validation Failed!**\n\nAvg Colors - R:{r_avg:.0f}, G:{g_avg:.0f}, B:{b_avg:.0f}\n\nThis doesn't appear to be a plant leaf!")
                st.info("Green should be dominant for leaves. Your image has red/skin tones.")

            # Image info
            st.markdown(
                f"""
                <div style='background: white; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <strong>Image Details:</strong><br>
                    📐 Size: {image.size[0]} x {image.size[1]} pixels<br>
                    🎨 Mode: {image.mode}<br>
                    📁 Format: {image.format}<br>
                    🎨 Avg RGB: ({r_avg:.0f}, {g_avg:.0f}, {b_avg:.0f})
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown("### 🔍 AI Analysis Results")

            # Show warning if not leaf color
            if not is_leaf_color:
                st.markdown(
                    """
                    <div class="prediction-box" style="background: linear-gradient(135deg, #D32F2F 0%, #F44336 100%);">
                        <div class="prediction-name">🚫 NOT A LEAF!</div>
                        <div class="confidence-score">Invalid Image</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.error(
                    """
                    ❌ **This is NOT a plant leaf!**

                    The image shows skin tones / human photo / non-leaf object.

                    **This app ONLY works with:**
                    - 🌿 GREEN plant leaves
                    - 🍂 BROWN dried leaves
                    - 📸 Close-up leaf photos

                    **Please upload an actual LEAF image!**
                    """
                )

                st.warning("If you still want to see prediction (for testing), check the box below:")
                force_predict = st.checkbox("Force prediction anyway (not recommended)")

                if not force_predict:
                    st.stop()




        with col2:


            # Add checkbox to bypass validation (for testing/advanced users)
            skip_validation = st.checkbox("Skip image validation (I confirm this is a leaf)", value=False)

            # STEP 1: Validate if image is actually a leaf
            if not skip_validation:
                is_valid, validation_message = is_valid_leaf_image(image)

                if not is_valid:
                    # Image doesn't look like a leaf!
                    st.markdown(
                        """
                        <div class="prediction-box" style="background: linear-gradient(135deg, #D32F2F 0%, #F44336 100%);">
                            <div class="prediction-name">🚫 Invalid Image</div>
                            <div class="confidence-score">Not a Leaf!</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.error(
                        f"""
                        ❌ **{validation_message}**

                        **This app can only identify medicinal plant LEAVES.**

                        ✅ **What to upload:**
                        - Close-up photo of a SINGLE LEAF
                        - GREEN or BROWN plant leaf
                        - Clear focus on the leaf
                        - Plain/simple background
                        - Good lighting (natural daylight)

                        ❌ **Do NOT upload:**
                        - Photos of PEOPLE, faces, hands, or body parts
                        - Photos of flowers, fruits, or stems
                        - Entire plant photos
                        - Random objects
                        - Very dark or blurry images

                        ---

                        **If you're SURE this is a leaf**, check the box above to bypass validation.
                        """
                    )

                    # Stop here - don't run prediction
                    st.stop()
            else:
                st.warning("⚠️ Validation bypassed - proceeding with prediction...")

            # STEP 2: If valid or bypassed, proceed with prediction
            with st.spinner("🔬 Analyzing leaf characteristics with AI..."):
                predictions = predict_plant(model, image)
                predicted_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_idx]
                confidence = predictions[predicted_idx] * 100

                # CONFIDENCE THRESHOLD - Important for unknown plants!
                CONFIDENCE_THRESHOLD = 60.0  # Adjust this value (40-70 recommended)

                # Check if this might be an unknown plant
                if confidence < CONFIDENCE_THRESHOLD:
                    # Unknown or out-of-dataset plant
                    st.markdown(
                        f"""
                        <div class="prediction-box" style="background: linear-gradient(135deg, #F57C00 0%, #FF9800 100%);">
                            <div class="prediction-name">⚠️ Unknown Plant</div>
                            <div class="confidence-score">Not in Database</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.error(
                        f"""
                        ❌ **Plant Not Recognized!**

                        This plant is likely **NOT in our database** of 30 medicinal plants.

                        **Possible reasons:**
                        - This plant is not one of the 30 species we can identify
                        - Image quality is too poor
                        - The leaf is damaged or unclear

                        **What you can do:**
                        1. Check if your plant is in the supported list (see sidebar)
                        2. Upload a clearer, better-lit image
                        3. Try a different leaf from the same plant

                        **Best match was:** {predicted_class} ({confidence:.2f}% confidence)
                        But this is **too low** to be reliable!
                        """
                    )

                    # Show what the model "guessed" anyway
                    with st.expander("🔍 See Model's Best Guess (Unreliable)"):
                        st.warning(
                            f"The model's closest match was **{predicted_class}** with only **{confidence:.2f}%** confidence. This is NOT reliable!")
                        st.info("Please only upload plants from our supported list for accurate results.")

                else:
                    # Plant is likely in database - show normal prediction
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <div class="prediction-name">🌿 {predicted_class}</div>
                            <div class="confidence-score">Confidence: {confidence:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Confidence indicator
                    if confidence > 90:
                        st.success("🎯 High confidence prediction! The model is very certain about this identification.")
                    elif confidence > 75:
                        st.success("✅ Good confidence! This prediction is reliable.")
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        st.warning(
                            "⚠️ Moderate confidence. The prediction might be correct, but consider uploading a clearer image for better accuracy.")

                # Additional check: if second-best prediction is very close
                sorted_indices = np.argsort(predictions)[::-1]
                second_best_confidence = predictions[sorted_indices[1]] * 100

                if confidence >= CONFIDENCE_THRESHOLD and (confidence - second_best_confidence) < 10:
                    st.info(
                        f"ℹ️ Note: The model also considered **{class_names[sorted_indices[1]]}** ({second_best_confidence:.2f}%). These plants may have similar leaf characteristics.")

        # Confidence chart
        st.markdown("---")

        # Only show detailed info if confidence is above threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            st.markdown("### 📊 Confidence Distribution")
            st.caption("Top 5 predictions with confidence scores")
            fig = create_confidence_chart(predictions, class_names)
            st.plotly_chart(fig, use_container_width=True)

            # Plant information with beautiful cards
            st.markdown("---")
            st.markdown("### 🌿 Detailed Plant Information")

        if predicted_class in PLANT_INFO:
            info = PLANT_INFO[predicted_class]

            # Create two columns for better layout
            info_col1, info_col2 = st.columns([1, 1])

            with info_col1:
                # Basic Information
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown('<p class="info-title">📋 Basic Information</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="scientific-name">Scientific Name: {info["scientific_name"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p class="info-content"><strong>🏷️ Common Names:</strong> {info["common_names"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p class="info-content"><strong>👨‍👩‍👧‍👦 Family:</strong> {info["family"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p class="info-content"><strong>🌿 Parts Used:</strong> {info["parts_used"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Preparation Methods
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown('<p class="info-title">🧪 Preparation Methods</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="info-content">{info["preparation"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with info_col2:
                # Description
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown('<p class="info-title">📖 Description</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="info-content">{info["description"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Precautions
                st.markdown('<div class="info-section">', unsafe_allow_html=True)
                st.markdown('<p class="info-title">⚠️ Precautions & Safety</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="info-content">{info["precautions"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Medicinal Uses - Full width
            st.markdown('<div class="info-section">', unsafe_allow_html=True)
            st.markdown('<p class="info-title">💊 Medicinal Uses & Health Benefits</p>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="info-content">{info["uses"]}</p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Text-to-Speech Section
            st.markdown("---")
            st.markdown("### 🔊 Listen to Information")
            st.caption("Get audio description of the identified plant")

            col_audio1, col_audio2 = st.columns([1, 3])

            with col_audio1:
                play_audio = st.button("🎧 Play Audio Description", use_container_width=True)

            with col_audio2:
                st.info("Click the button to hear a detailed description of this medicinal plant")

            if play_audio:
                speech_text = f"""
                {predicted_class}. Scientific name: {info['scientific_name']}. 
                Common names include: {info['common_names']}. 
                {info['description']} 
                The medicinal uses include: {info['uses']}. 
                Parts commonly used are: {info['parts_used']}.
                """

                with st.spinner("🎵 Generating audio..."):
                    audio_file = text_to_speech(speech_text, predicted_class)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                        os.unlink(audio_file)
                        st.success("✅ Audio generated successfully!")

        else:
            # For unknown plants, show alternative actions
            st.markdown("---")
            st.markdown("### 💡 What to Do Next?")

            col_help1, col_help2 = st.columns(2)

            with col_help1:
                st.markdown(
                    """
                    <div class="info-section">
                        <h4>✅ Verify Your Plant</h4>
                        <p>Check the sidebar to see all 30 plants we can identify. Make sure your plant is in the list!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_help2:
                st.markdown(
                    """
                    <div class="info-section">
                        <h4>📸 Improve Image Quality</h4>
                        <p>Take a new photo with better lighting, clear focus, and plain background for more accurate results.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Show top 5 anyway for reference
            st.markdown("### 📊 Top 5 Closest Matches (For Reference Only)")
            st.warning("⚠️ These predictions are unreliable due to low confidence!")
            fig = create_confidence_chart(predictions, class_names)
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Beautiful welcome screen
        st.markdown(
            """
            <div style='text-align: center; padding: 3rem; background: white; border-radius: 20px; margin: 2rem 0;'>
                <h2 style='color: #2E7D32; margin-bottom: 1rem;'>👆 Upload a Leaf Image to Get Started</h2>
                <p style='font-size: 1.2rem; color: #666;'>Our AI will identify the medicinal plant and provide detailed information</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tips section with beautiful styling
        st.markdown(
            """
            <div class="tips-box">
                <h3>💡 Tips for Best Results</h3>
                <ul>
                    <li>Use clear, well-lit images</li>
                    <li>Ensure the leaf is in focus</li>
                    <li>Capture the entire leaf if possible</li>
                    <li>Avoid blurry or dark images</li>
                    <li>Use a plain background for better recognition</li>
                    <li>Take photos from the top view of the leaf</li>
                    <li>Avoid shadows on the leaf</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Sample showcase
        st.markdown("### 🖼️ Sample Plants in Database")
        st.caption("Examples of medicinal plants our AI can identify")

        sample_col1, sample_col2, sample_col3 = st.columns(3)

        with sample_col1:
            st.markdown(
                """
                <div class="info-section" style="text-align: center;">
                    <h4>🌿 Tulsi</h4>
                    <p>Holy Basil - Queen of Herbs</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with sample_col2:
            st.markdown(
                """
                <div class="info-section" style="text-align: center;">
                    <h4>🌿 Neem</h4>
                    <p>Divine Tree - Nature's Pharmacy</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with sample_col3:
            st.markdown(
                """
                <div class="info-section" style="text-align: center;">
                    <h4>🌿 Curry</h4>
                    <p>Curry Leaf - Aromatic Medicine</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; color: #2E7D32;'>
            <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
                <strong>🌿 Medicinal Plant Classifier</strong>
            </p>
            <p style='font-size: 0.9rem;'>
                Powered by Deep Learning | Accuracy: 97.59% | 30 Plant Species
            </p>
            <p style='font-size: 0.8rem; color: #666; margin-top: 1rem;'>
                ⚠️ Disclaimer: This tool is for educational purposes only. 
                Always consult healthcare professionals before using any medicinal plants.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()