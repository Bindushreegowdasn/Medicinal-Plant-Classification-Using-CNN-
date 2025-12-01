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

# Import the chatbot
from chatbot_engine import MedicinalPlantChatbot

# Page configuration
st.set_page_config(
    page_title="Medicinal Plant Classifier with Chatbot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED FOR FULL VISIBILITY
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);
    }

    .main-header {
        font-size: 3.5rem;
        color: #1B5E20;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.15);
    }

    .sub-header {
        font-size: 1.3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-box {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(67, 160, 71, 0.4);
    }

    .prediction-name {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .confidence-score {
        font-size: 1.8rem;
        opacity: 0.95;
        font-weight: 500;
    }

    .warning-box {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
    }

    .info-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .info-title {
        font-size: 1.6rem;
        color: #2E7D32;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid #66BB6A;
        padding-bottom: 0.5rem;
    }

    /* FIXED CHAT CONTAINER - REMOVED max-height LIMIT */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        overflow-y: auto;
        min-height: 300px;
    }

    .user-message {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        color: #1B5E20;
        font-weight: 500;
        word-wrap: break-word;
    }

    .bot-message {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        color: white;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

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
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #C8E6C9 0%, #A5D6A7 100%);
    }

    /* ENSURE RESPONSIVE LAYOUT */
    .stColumn {
        padding: 0.5rem;
    }

    /* FIX FOR PLANT INFO SECTIONS */
    .element-container {
        margin-bottom: 1rem;
    }

    /* ENSURE TEXT IS FULLY VISIBLE */
    p, div, span {
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

# PLANT_INFO DATABASE
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

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = MedicinalPlantChatbot(PLANT_INFO)
    st.session_state.chat_history = [{'bot': "Hello! 🌿 I'm your medicinal plant assistant. Ask me anything!"}]
    st.session_state.last_user_input = ""
    st.session_state.auto_clear = True


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('models/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def check_if_plant_image(image):
    """
    Check if the image likely contains a plant/leaf
    Returns: (is_plant, reason)
    """
    img_array = np.array(image.convert('RGB'))

    # Check 1: Green pixel ratio (plants are usually green)
    hsv_img = tf.image.rgb_to_hsv(img_array / 255.0).numpy()

    # Define green hue range (roughly 60-180 degrees in HSV)
    green_hue_min = 0.16  # ~60 degrees
    green_hue_max = 0.5  # ~180 degrees

    green_mask = (hsv_img[:, :, 0] >= green_hue_min) & (hsv_img[:, :, 0] <= green_hue_max)
    green_mask = green_mask & (hsv_img[:, :, 1] > 0.2)  # Sufficient saturation

    green_ratio = np.sum(green_mask) / green_mask.size

    # Check 2: Skin tone detection (to reject human images)
    skin_hue_min = 0.0
    skin_hue_max = 0.1  # ~0-36 degrees (orange-ish)

    skin_mask = (hsv_img[:, :, 0] >= skin_hue_min) & (hsv_img[:, :, 0] <= skin_hue_max)
    skin_mask = skin_mask & (hsv_img[:, :, 1] > 0.15) & (hsv_img[:, :, 1] < 0.7)

    skin_ratio = np.sum(skin_mask) / skin_mask.size

    # Decision logic
    if skin_ratio > 0.15:  # More than 15% skin tones
        return False, "This appears to be a photo of a person, not a plant. Please upload a clear image of a plant leaf."

    if green_ratio < 0.10:  # Less than 10% green
        return False, "This image doesn't appear to contain a plant. Please upload a clear leaf image with visible green color."

    return True, "Image validation passed"


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_plant(model, image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    return predictions[0]


def create_confidence_chart(predictions, class_names):
    """Create confidence chart"""
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    top_5_classes = [class_names[i] for i in top_5_idx]
    top_5_scores = [predictions[i] * 100 for i in top_5_idx]
    colors = ['#43A047' if i == 0 else '#81C784' for i in range(5)]

    fig = go.Figure(data=[
        go.Bar(
            x=top_5_scores,
            y=top_5_classes,
            orientation='h',
            marker=dict(color=colors, line=dict(color='#2E7D32', width=2)),
            text=[f'{score:.2f}%' for score in top_5_scores],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence (%)",
        height=400,
        showlegend=False,
    )
    return fig


def render_chatbot():
    """Render chatbot interface with auto-clear"""
    st.markdown("### 💬 Chat with Plant Expert")

    col_toggle, col_info = st.columns([1, 3])
    with col_toggle:
        st.session_state.auto_clear = st.toggle("Auto-Clear Mode", value=st.session_state.auto_clear,
                                                key="toggle_autoclear")
    with col_info:
        if st.session_state.auto_clear:
            st.caption("✨ Each new question starts a fresh conversation")
        else:
            st.caption("📜 Chat history will be preserved")

    chat_html = '<div class="chat-container">'
    for message in st.session_state.chat_history:
        if 'user' in message:
            chat_html += f'<div class="user-message">👤 {message["user"]}</div>'
        elif 'bot' in message:
            chat_html += f'<div class="bot-message">🤖 {message["bot"]}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    st.markdown("#### 🚀 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📋 List All Plants", key="btn_list"):
            process_chat("list all plants")
    with col2:
        if st.button("❓ How to Use", key="btn_help"):
            process_chat("help")
    with col3:
        if st.button("🌿 About Tulsi", key="btn_tulsi"):
            process_chat("tell me about tulsi")
    with col4:
        if st.button("💊 Diabetes Plants", key="btn_diabetes"):
            process_chat("plants for diabetes")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("🤧 Cold & Cough", key="btn_cold"):
            process_chat("plants for cold and cough")
    with col6:
        if st.button("💆 Headache Relief", key="btn_headache"):
            process_chat("plants for headache")
    with col7:
        if st.button("💪 Immunity Boost", key="btn_immunity"):
            process_chat("plants for immunity")
    with col8:
        if st.button("🌺 Skin Care", key="btn_skin"):
            process_chat("plants for skin")

    st.markdown("---")
    st.markdown("#### 💬 Ask Your Question")

    user_input = st.text_input(
        "Type here:",
        placeholder="E.g., 'What is Neem good for?' or 'Plants for digestion'",
        label_visibility="collapsed",
        key="user_input_main"
    )

    col_send, col_manual_clear = st.columns([4, 1])

    with col_send:
        if st.button("Send 📤", use_container_width=True, type="primary", key="btn_send_main"):
            if user_input and user_input.strip():
                process_chat(user_input)

    with col_manual_clear:
        if st.button("🔄 Reset", use_container_width=True, key="btn_manual_clear"):
            st.session_state.chat_history = [{'bot': "Chat reset! 🌿 What would you like to know?"}]
            st.session_state.chatbot.clear_history()
            st.session_state.last_user_input = ""
            st.rerun()


def process_chat(user_input):
    """Process chat input with auto-clear option"""
    if user_input.strip():
        if user_input == st.session_state.last_user_input:
            return

        st.session_state.last_user_input = user_input

        if st.session_state.auto_clear:
            st.session_state.chat_history = []
            st.session_state.chatbot.clear_history()

        st.session_state.chat_history.append({'user': user_input})
        bot_response = st.session_state.chatbot.chat(user_input)
        st.session_state.chat_history.append({'bot': bot_response})
        st.rerun()


def main():
    st.markdown('<p class="main-header">🌿 Medicinal Plant AI Classifier + Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify plants by image & chat with our AI expert!</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("📚 About")
        st.info(
            "🎯 **Accuracy:** 97.59%\n\n🧠 **AI Model:** CNN\n\n🌍 **Plants:** 30 Species\n\n💬 **Chatbot:** Rule-based NLP")

        st.markdown("---")
        st.subheader("🌱 Features")
        st.markdown("✅ Image Classification\n✅ Plant Validation\n✅ AI Chatbot\n✅ Plant Information")

        st.markdown("---")
        st.subheader("📸 Tips for Best Results")
        st.markdown("""
        • Use clear, well-lit images
        • Capture the full leaf
        • Avoid blurry photos
        • Only upload plant leaves
        • Avoid human/animal images
        """)

    tab1, tab2 = st.tabs(["🔬 Image Classification", "💬 Chatbot"])

    with tab1:
        st.markdown("### 📤 Upload Leaf Image")

        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file:
            model = load_model()
            if model:
                image = Image.open(uploaded_file)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image, caption="Uploaded Image", width=300)

                with col2:
                    with st.spinner("Validating image..."):
                        # CRITICAL: Check if it's a plant image first
                        is_plant, reason = check_if_plant_image(image)

                        if not is_plant:
                            st.markdown(
                                f"""
                                <div class="warning-box">
                                    <h2>⚠️ Invalid Image</h2>
                                    <p style="font-size: 1.2rem;">{reason}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.warning("**Please upload a clear image of a plant leaf for classification.**")
                        else:
                            # Proceed with classification
                            with st.spinner("Analyzing plant..."):
                                class_names = sorted(PLANT_INFO.keys())
                                predictions = predict_plant(model, image)
                                predicted_idx = np.argmax(predictions)
                                predicted_class = class_names[predicted_idx]
                                confidence = predictions[predicted_idx] * 100

                                # Add confidence threshold check
                                if confidence < 70:
                                    st.markdown(
                                        f"""
                                        <div class="warning-box">
                                            <h2>⚠️ Low Confidence</h2>
                                            <p style="font-size: 1.2rem;">The model is only {confidence:.2f}% confident.</p>
                                            <p>Please try uploading a clearer image of the plant leaf.</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"""
                                        <div class="prediction-box">
                                            <div class="prediction-name">🌿 {predicted_class}</div>
                                            <div class="confidence-score">Confidence: {confidence:.2f}%</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                # Show chart in col2
                with col2:
                    st.plotly_chart(create_confidence_chart(predictions, class_names), use_container_width=True)

                # Show plant info only if confidence is good - FULL WIDTH BELOW
                if confidence >= 70 and predicted_class in PLANT_INFO:
                    info = PLANT_INFO[predicted_class]

                    # FULL WIDTH LAYOUT WITH TWO COLUMNS
                    st.markdown("---")
                    st.markdown(f"### 🌿 About {predicted_class}")

                    # Create two columns for better left-to-right layout
                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        with st.expander("📖 Scientific Information", expanded=True):
                            st.markdown(f"**Scientific Name:** *{info['scientific_name']}*")
                            st.markdown(f"**Common Names:** {info['common_names']}")
                            if 'family' in info:
                                st.markdown(f"**Family:** {info['family']}")

                        with st.expander("💊 Medicinal Uses", expanded=True):
                            st.markdown(f"**Uses:** {info['uses']}")
                            if 'description' in info:
                                st.markdown(f"**Description:** {info['description']}")

                    with info_col2:
                        with st.expander("🔬 Usage Information", expanded=True):
                            if 'parts_used' in info:
                                st.markdown(f"**Parts Used:** {info['parts_used']}")
                            st.markdown(f"**Preparation:** {info['preparation']}")

                        with st.expander("⚠️ Precautions", expanded=True):
                            st.warning(f"**{info['precautions']}**")

        else:
            st.info("👆 Upload a **plant leaf image** to get started!")
            st.markdown("""
            **Guidelines:**
            - ✅ Clear images of plant leaves
            - ✅ Good lighting conditions
            - ✅ Focus on the leaf structure
            - ❌ No human faces or body parts
            - ❌ No animals or objects
            """)

    with tab2:
        render_chatbot()

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #2E7D32;'>🌿 Medicinal Plant Classifier with AI Chatbot | Educational Tool</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()