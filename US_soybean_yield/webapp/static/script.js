window.onload = function () {
    document.getElementById("submit").addEventListener('click', async (event) => {
        event.preventDefault();
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                lat: document.getElementById("lat").value,
                lon: document.getElementById("lon").value,
                year: document.getElementById("year").value,
                JJA_SWdown: document.getElementById("JJA_SWdown").value,
                JJA_LWnet: document.getElementById("JJA_LWnet").value,
                JFMA_EF: document.getElementById("JFMA_EF").value,
                JJA_EF: document.getElementById("JJA_EF").value,
                SO_EF: document.getElementById("SO_EF").value,
                JFMA_Rainf: document.getElementById("JFMA_Rainf").value,
                MJJ_Rainf: document.getElementById("MJJ_Rainf").value,
                JF_Snowf: document.getElementById("JF_Snowf").value,
                JJA_ESoil: document.getElementById("JJA_ESoil").value,
                MJJA_Albedo: document.getElementById("MJJA_Albedo").value,
                MJJA_SoilM_0_10cm: document.getElementById("MJJA_SoilM_0_10cm").value,
                JFMA_RootMoist: document.getElementById("JFMA_RootMoist").value,
                JanToApr_LAI: document.getElementById("JanToApr_LAI").value,
                MayToOct_ACond: document.getElementById("MayToOct_ACond").value,
                Lead_AnnualTotal_Rainf: document.getElementById("Lead_AnnualTotal_Rainf").value,
                ECanop_Jan: document.getElementById("ECanop_Jan").value,
                ACond_Jan: document.getElementById("ACond_Jan").value,
                Qle_Jan: document.getElementById("Qle_Jan").value,
                GVEG_Apr: document.getElementById("GVEG_Apr").value,
                SoilM_100_200cm_Sep: document.getElementById("SoilM_100_200cm_Sep").value,
                LWnet_May: document.getElementById("LWnet_May").value,
                ESoil_May: document.getElementById("ESoil_May").value,
                AvgSurfT_Aug: document.getElementById("AvgSurfT_Aug").value,
            })
        });
        const json = await response.json();
        document.getElementById("result").innerHTML = `Predicted Yield (t ha-1): ${json.Predicted_Yield}`;
    });
};