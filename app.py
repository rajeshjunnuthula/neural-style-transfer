import streamlit as st
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title(" Neural Style Transfer (Optimized VGG)")
st.write("Faster classical Neural Style Transfer using optimized VGG layers.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- UI ----------------
content_file = st.file_uploader(" Upload Content Image", ["jpg", "jpeg", "png"])
style_file = st.file_uploader(" Upload Style Image", ["jpg", "jpeg", "png"])
steps = st.slider(" Optimization Steps (Lower = Faster)", 30, 120, 60)

st.divider()

# ---------------- HELPERS ----------------
def load_image(image, max_size=256):  #  Reduced size
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# ---------------- MAIN LOGIC ----------------
if content_file and style_file and st.button(" Generate Stylized Image"):

    with st.spinner("Loading images..."):
        content_img = Image.open(content_file).convert("RGB")
        style_img = Image.open(style_file).convert("RGB")

        st.image([content_img, style_img], caption=["Content", "Style"], width=250)

        content = load_image(content_img)
        style = load_image(style_img)

    with st.spinner("Loading optimized VGG19 model..."):
        vgg = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT
        ).features.to(device).eval()

    # Fewer layers = faster
    style_layers = ['0', '5', '10']
    content_layer = '10'

    generated = content.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=0.003)

    with st.spinner("Applying style (≈1–3 minutes)..."):
        for _ in range(steps):
            gen_features = {}
            content_features = {}
            style_features = {}

            x = generated
            c = content
            s = style

            for name, layer in vgg._modules.items():
                x = layer(x)
                c = layer(c)
                s = layer(s)

                if name == content_layer:
                    content_features[name] = c
                    gen_features[name] = x

                if name in style_layers:
                    style_features[name] = s
                    gen_features[name] = x

            content_loss = torch.mean(
                (gen_features[content_layer] - content_features[content_layer]) ** 2
            )

            style_loss = 0
            for layer in style_layers:
                g_gram = gram_matrix(gen_features[layer])
                s_gram = gram_matrix(style_features[layer])
                style_loss += torch.mean((g_gram - s_gram) ** 2)

            total_loss = content_loss + 1e6 * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    output = generated.squeeze().detach().cpu().clamp(0, 1)

    st.success("Stylization Complete")
    st.image(
        output.permute(1, 2, 0).numpy(),
        caption="Stylized Output"
    )

else:
    st.info("Upload both images and click **Generate Stylized Image**")