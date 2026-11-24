import folium
import pandas as pd
import geopandas as gpd


class GISVisualizerAndAllocator:
    """Renders interactive Folium priority maps and computes proportional relief allocation."""

    TIER_COLORS = {
        "Critical": "#99000d",
        "High": "#ef3b2c",
        "Moderate": "#fc9272",
    }

    def allocate_relief_kits(self, df_wards: pd.DataFrame, total_kits: int) -> pd.DataFrame:
        out = df_wards.sort_values("Vulnerability_Score", ascending=False).reset_index(drop=True)
        out["Weighted_Need"] = out["Pop"] * out["Vulnerability_Score"]
        out["Allocated_Kits"] = (
            (out["Weighted_Need"] / out["Weighted_Need"].sum()) * total_kits
        ).round().astype(int)
        return out

    def build_map(self, df_wards: pd.DataFrame, flood_gdf: gpd.GeoDataFrame) -> folium.Map:
        m = folium.Map(location=[-1.55, 39.95], zoom_start=8, tiles="OpenStreetMap")

        folium.GeoJson(
            flood_gdf,
            name="Historical Flood Inundation Zones",
            style_function=lambda _: {
                "fillColor": "#1f78b4",
                "color": "#1f78b4",
                "weight": 1,
                "fillOpacity": 0.25,
            },
            tooltip=folium.GeoJsonTooltip(fields=["inund_hist"], aliases=["Flood Depth (m):"]),
        ).add_to(m)

        for _, row in df_wards.iterrows():
            tier = str(row["Priority_Tier"])
            color = self.TIER_COLORS.get(tier, "#ef3b2c")
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.5;">
                <b>Ward:</b> {row['Ward']} ({row['Sub_County']})<br>
                <b>Priority Tier:</b> <span style='color:{color};'><b>{tier}</b></span><br>
                <b>Vulnerability Score:</b> {row['Vulnerability_Score']:.4f}<br>
                <b>Allocated Relief Kits:</b> {row['Allocated_Kits']:,}<br>
                <b>Poverty Rate:</b> {row['Poverty_Rate']}% | <b>Flood Depth:</b> {row['Flood_Inundation_m']} m
            </div>
            """
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=13 if tier == "Critical" else (9 if tier == "High" else 7),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.88,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{row['Ward']} | Score: {row['Vulnerability_Score']:.3f} ({tier})",
            ).add_to(m)

        legend_html = """
        <div style="position: fixed; bottom: 25px; left: 25px; width: 215px;
                    background-color: white; border: 1px solid #cbd5e1; z-index: 9999;
                    font-family: Arial, sans-serif; font-size: 12px; padding: 10px 12px;
                    border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.18);">
            <b style="color:#1a1f2b;">Vulnerability Priority Tier</b><br>
            <div style="margin-top:4px;">
                <span style="color:#99000d; font-size:15px;">●</span> Critical Priority<br>
                <span style="color:#ef3b2c; font-size:13px;">●</span> High Priority<br>
                <span style="color:#fc9272; font-size:11px;">●</span> Moderate Priority
            </div>
            <hr style="margin: 6px 0; border-top: 1px solid #e2e8f0;">
            <span style="color:#1f78b4;">■</span> Flood Inundation Zone
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
        return m