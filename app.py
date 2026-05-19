import dash
from dash import dcc, html, Input, Output, State, ctx, no_update, Patch
import plotly.graph_objects as go
import numpy as np
import paho.mqtt.client as mqtt
import json
import ssl
import boto3
import threading
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Key, Attr

# ==========================================
# 1. CONFIGURACIÓ AWS
# ==========================================
ENDPOINT  = "a3w35veduf1p9-ats.iot.eu-north-1.amazonaws.com"
CA_PATH   = "AmazonRootCA1.pem"
CERT_PATH = "306dce26e528b14a881c78af0efcd0e046210e610dd09541fa2e5604d2a26b35-certificate.pem.crt"
KEY_PATH  = "306dce26e528b14a881c78af0efcd0e046210e610dd09541fa2e5604d2a26b35-private.pem.key"

dynamodb       = boto3.resource('dynamodb', region_name='us-east-1')
taula_sessions = dynamodb.Table('SessionsCoixi')
taula_usuaris  = dynamodb.Table('Usuaris_Coixi')

dades_globals     = np.zeros((24, 24))
smooth_globals    = np.zeros((24, 24))
N_FIL, N_COL      = 24, 24
last_mqtt_message = None
mqtt_connected    = False
current_topic     = None
mode_global = -1

# ==========================================
# 2. MQTT
# ==========================================
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="Python_TFG_Final")
mqtt_client.tls_set(CA_PATH, certfile=CERT_PATH, keyfile=KEY_PATH,
                    cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)

def on_connect(client, userdata, flags, rc):
    global mqtt_connected, current_topic
    mqtt_connected = (rc == 0)
    if mqtt_connected and current_topic:
        client.subscribe(current_topic)

def on_disconnect(client, userdata, rc):
    global mqtt_connected
    if rc != 0:  # només marca desconnectat si no és voluntari
        mqtt_connected = False

def on_message(client, userdata, msg):
    global dades_globals, smooth_globals, last_mqtt_message, mode_global
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        if 'matriu' in payload:
            raw = np.array(payload['matriu']).reshape(24, 24).astype(float)
            mode_global = payload.get('mode', -1)  # ← guarda el mode
            smooth_globals    = 0.8 * smooth_globals + 0.2 * raw  # EMA α=0.2
            dades_globals     = smooth_globals.copy()
            last_mqtt_message = datetime.now()
    except Exception:
        pass

mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message    = on_message

# ==========================================
# 3. DYNAMO ASYNC — no bloqueja el render
# ==========================================
def guardar_a_dynamo_async(uid, timestamp, matriu, notes, nom_sessio):
    try:
        mat      = np.array(matriu).reshape(24, 24)  
        max_val  = int(np.max(mat))
        avg_val  = float(np.mean(mat[mat > 50])) if np.any(mat > 50) else 0.0
        n_active = int(np.sum(mat > 50))

        taula_sessions.put_item(Item={
            'PatientID':    uid,
            'Timestamp':    timestamp,
            'NomSessio':    nom_sessio or 'Sessió sense nom',
            'Matriu':       json.dumps(matriu),
            'Observacions': notes or 'N/A',
            'MaxVal':       str(max_val),
            'AvgVal':       str(round(avg_val, 1)),
            'NCeles':       str(n_active),
        })
    except Exception as e:
        print(f"[DynamoDB] Error: {e}")

# ==========================================
# 4. COLORSCALE I FILTRES VISUALS
# ==========================================
HEATMAP_COLORSCALE = [
    [0.000, 'rgb(0,0,20)'],
    [0.001, 'rgb(0,0,200)'],
    [0.250, 'rgb(0,255,255)'],
    [0.500, 'rgb(0,255,0)'],
    [0.750, 'rgb(255,255,0)'],
    [1.000, 'rgb(255,0,0)'],
]

def apply_heatmap_filters(mat, threshold=50, gamma=1.5):
    mat     = mat.astype(float)
    max_val = np.max(mat)
    if max_val <= threshold:
        return np.zeros_like(mat)
    mat_norm = np.where(mat >= threshold,
                        (mat - threshold) / (max_val - threshold), 0.0)
    return np.power(np.clip(mat_norm, 0, 1), gamma)

# ==========================================
# 5. FIGURES
# ==========================================
def build_empty_figure():
    """Figura base carregada una sola vegada. Patch actualitza només z."""
    fig = go.Figure(data=go.Heatmap(
        z=np.zeros((N_FIL, N_COL)).tolist(),
        colorscale=HEATMAP_COLORSCALE,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='Intensitat', font=dict(size=11, color='#94a3b8')),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=['0', '25%', '50%', '75%', '100%'],
            tickfont=dict(size=10, color='#94a3b8'),
            thickness=12, len=0.85,
            bgcolor='rgba(0,0,0,0)', outlinewidth=0,
        ),
        hoverongaps=False,
        hovertemplate='Fila %{y} · Col %{x}<br>Valor: %{customdata} ADC<extra></extra>',
        customdata=np.zeros((N_FIL, N_COL), dtype=int).tolist(),
        x=list(range(N_COL)),
        y=list(range(N_FIL)),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        uirevision='coixi',
        margin=dict(l=0, r=50, t=40, b=0),
        paper_bgcolor='rgba(11,18,32,0)',
        plot_bgcolor='rgba(11,18,32,1)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True, range=[-0.5, N_COL - 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True, autorange='reversed',
                   range=[-0.5, N_FIL - 0.5], scaleanchor='x'),
        annotations=[dict(xref='paper', yref='paper', x=0.0, y=1.04,
                          text='<b>Màxim: 0 ADC</b>', showarrow=False,
                          font=dict(size=13, color='#f87171'))],
        dragmode=False,
    )
    return fig

def build_heatmap_figure(mat):
    """Figura completa per a l'historial (no es construeix en temps real)."""
    mat_norm = apply_heatmap_filters(mat)
    max_val  = int(np.max(mat))
    fig = go.Figure(data=go.Heatmap(
        z=mat_norm.tolist(),
        colorscale=HEATMAP_COLORSCALE,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text='Intensitat', font=dict(size=11, color='#94a3b8')),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=['0', '25%', '50%', '75%', '100%'],
            tickfont=dict(size=10, color='#94a3b8'),
            thickness=12, len=0.85,
            bgcolor='rgba(0,0,0,0)', outlinewidth=0,
        ),
        hoverongaps=False,
        hovertemplate='Fila %{y} · Col %{x}<br>Valor: %{customdata} ADC<extra></extra>',
        customdata=mat.astype(int).tolist(),
        x=list(range(N_COL)),
        y=list(range(N_FIL)),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        margin=dict(l=0, r=50, t=40, b=0),
        paper_bgcolor='rgba(11,18,32,0)',
        plot_bgcolor='rgba(11,18,32,1)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True, range=[-0.5, N_COL - 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   fixedrange=True, autorange='reversed',
                   range=[-0.5, N_FIL - 0.5], scaleanchor='x'),
        annotations=[dict(xref='paper', yref='paper', x=0.0, y=1.04,
                          text=f'<b>Màxim: {max_val} ADC</b>', showarrow=False,
                          font=dict(size=13, color='#f87171'))],
        dragmode=False,
    )
    return fig

# ==========================================
# 6. ESTILS
# ==========================================
FONT      = "'DM Mono', 'Courier New', monospace"
FONT_SANS = "'DM Sans', 'Segoe UI', sans-serif"

C = {
    'bg':      '#080e1a', 'surface': '#0f1929', 'panel':   '#111e30',
    'border':  '#1e3048', 'accent':  '#3b82f6', 'accent2': '#06b6d4',
    'danger':  '#f87171', 'success': '#34d399', 'warning': '#fbbf24',
    'text':    '#e2e8f0', 'muted':   '#64748b', 'sidebar': '#0a1220',
}

CARD = {
    'backgroundColor': C['panel'], 'borderRadius': '16px', 'padding': '20px',
    'border': f"1px solid {C['border']}", 'boxShadow': '0 4px 24px rgba(0,0,0,0.4)',
}

INPUT_STYLE = {
    'width': '100%', 'padding': '11px 14px', 'borderRadius': '10px',
    'border': f"1px solid {C['border']}", 'outline': 'none',
    'fontSize': '14px', 'fontFamily': FONT_SANS,
    'backgroundColor': C['surface'], 'color': C['text'], 'boxSizing': 'border-box',
}

BTN = {
    'width': '100%', 'padding': '12px 16px', 'borderRadius': '12px',
    'fontWeight': '700', 'border': 'none', 'cursor': 'pointer',
    'fontSize': '13px', 'fontFamily': FONT_SANS,
    'letterSpacing': '0.5px', 'transition': 'all 0.2s ease',
}

# ==========================================
# 7. APP
# ==========================================
app       = dash.Dash(__name__, suppress_callback_exceptions=True)
server    = app.server
app.title = "R+ Pressió"

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #080e1a; color: #e2e8f0; font-family: 'DM Sans', sans-serif; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #080e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3048; border-radius: 3px; }
  /* ── MÒBIL ─────────────────────────────────── */
  @media (max-width: 768px) {

    /* Sidebar ocult per defecte en mòbil */
    #sidebar-panel {
      position: fixed !important;
      top: 64px; left: 0; bottom: 0;
      z-index: 999;
      width: 280px !important;
      min-width: 280px !important;
      transform: translateX(-100%);
      transition: transform 0.25s ease;
      overflow-y: auto !important;
    }
    #sidebar-panel.open {
      transform: translateX(0);
    }
    #content-panel {
      width: 100% !important;
      padding: 8px !important;
      min-width: 0 !important;
    }
    #kpi-max-val, #kpi-avg-val, #kpi-state-val, #kpi-mode-val {
      font-size: 18px !important;
      word-break: break-word !important;
      overflow: hidden !important;
    }
    .kpi-grid {
      grid-template-columns: repeat(2, 1fr) !important;
    }
    .js-plotly-plot {
      height: 45vh !important;
      min-height: 300px !important;
    }
  }
</style>
<script>
  document.addEventListener('click', function(e) {
    if (e.target.id === 'btn-sidebar' || e.target.closest('#btn-sidebar')) {
      var sidebar = document.getElementById('sidebar-panel');
      if (sidebar) sidebar.classList.toggle('open');
    }
  });
</script>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>'''

# ==========================================
# 8. LOGIN
# ==========================================
layout_login = html.Div([
    html.Div([
        html.Div([
            html.Div("R+", style={
                'fontSize': '72px', 'fontWeight': '900', 'fontFamily': FONT_SANS,
                'background': f'linear-gradient(135deg,{C["accent"]},{C["accent2"]})',
                'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
                'lineHeight': '1',
            }),
            html.Div("Sistema de monitorització de pressió", style={
                'fontSize': '14px', 'color': C['muted'], 'marginTop': '8px'
            }),
        ], style={'marginBottom': '36px', 'textAlign': 'center'}),

        html.Div([
            html.Label('Usuari', style={
                'fontWeight': '700', 'fontSize': '12px', 'color': C['muted'],
                'marginBottom': '6px', 'display': 'block',
                'textTransform': 'uppercase', 'letterSpacing': '1px',
            }),
            dcc.Input(id='user-in', type='text', placeholder="nom d'usuari",
                      style=INPUT_STYLE)
        ], style={'marginBottom': '16px'}),

        html.Div([
            html.Label('Contrasenya', style={
                'fontWeight': '700', 'fontSize': '12px', 'color': C['muted'],
                'marginBottom': '6px', 'display': 'block',
                'textTransform': 'uppercase', 'letterSpacing': '1px',
            }),
            dcc.Input(id='pass-in', type='password', placeholder='••••••••',
                      style=INPUT_STYLE)
        ], style={'marginBottom': '24px'}),

        html.Button('Accedir al panell', id='btn-login', n_clicks=0, style={
            **BTN,
            'background': f'linear-gradient(135deg,{C["accent"]},{C["accent2"]})',
            'color': 'white', 'boxShadow': '0 8px 32px rgba(59,130,246,0.35)',
        }),
        html.Div(id='login-msg', style={
            'marginTop': '16px', 'fontSize': '13px',
            'minHeight': '20px', 'textAlign': 'center', 'fontFamily': FONT,
        })
    ], style={
        'width': '100%', 'maxWidth': '400px', 'backgroundColor': C['panel'],
        'borderRadius': '24px', 'padding': '40px',
        'border': f"1px solid {C['border']}", 'boxShadow': '0 32px 80px rgba(0,0,0,0.6)',
    })
], style={
    'minHeight': '100vh', 'display': 'flex', 'alignItems': 'center',
    'justifyContent': 'center', 'padding': '24px',
    'background': f'radial-gradient(ellipse at 30% 20%, rgba(59,130,246,0.12) 0%, {C["bg"]} 60%)',
})

# ==========================================
# 9. COMPONENTS
# ==========================================
def kpi_card(title, value_id, subtitle_id=None, accent=C['accent']):
    return html.Div([
        html.Div(title, style={
            'fontSize': '11px', 'color': C['muted'], 'textTransform': 'uppercase',
            'letterSpacing': '1.5px', 'marginBottom': '10px', 'fontWeight': '700',
        }),
        html.Div(id=value_id, style={
            'fontSize': '28px', 'fontWeight': '900',
            'color': accent, 'fontFamily': FONT, 'lineHeight': '1',
        }),
        html.Div(id=subtitle_id, style={
            'fontSize': '12px', 'color': C['muted'], 'marginTop': '6px',
        }) if subtitle_id else html.Div(),
    ], style=CARD)

def status_dot(label, comp_id):
    return html.Div([
        html.Div([
            html.Div(id=f'{comp_id}-dot', style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%',
                'backgroundColor': C['muted'], 'flexShrink': '0',
            }),
            html.Div(label, style={'fontSize': '12px', 'color': C['muted']}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                  'marginBottom': '4px'}),
        html.Div(id=comp_id, style={
            'fontSize': '12px', 'color': C['text'],
            'fontFamily': FONT, 'paddingLeft': '16px',
        }),
    ], style={'marginBottom': '12px'})

# ==========================================
# 10. DASHBOARD
# ==========================================
def layout_dashboard(user_id, device_id):
    return html.Div([
        dcc.Store(id='store-user', data={'uid': user_id, 'did': device_id}),
        dcc.Store(id='store-rec',  data={'recording': False}),
        dcc.Store(id='store-ui',   data={'sidebar_open': True}),

        # HEADER
        html.Div([
            html.Div([
                html.Button('☰', id='btn-sidebar', n_clicks=0, style={
                    'border': f"1px solid {C['border']}", 'backgroundColor': C['surface'],
                    'color': C['text'], 'borderRadius': '10px', 'height': '38px',
                    'width': '38px', 'cursor': 'pointer', 'fontSize': '16px',
                    'marginRight': '14px', 'flexShrink': '0',
                }),
                html.Span("R+", style={
                    'fontSize': '22px', 'fontWeight': '900',
                    'background': f'linear-gradient(135deg,{C["accent"]},{C["accent2"]})',
                    'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
                }),
                html.Span(" Monitorització de pressió", style={
                    'fontSize': '13px', 'color': C['muted'], 'marginLeft': '8px',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Div(id='live-clock', style={
                    'fontSize': '13px', 'color': C['muted'],
                    'fontFamily': FONT, 'textAlign': 'right',
                }),
                html.Div(f'👤 {user_id}  ·  {device_id}', style={
                    'fontSize': '12px', 'color': C['muted'],
                    'textAlign': 'right', 'marginTop': '2px',
                }),
            ]),
        ], style={
            'height': '64px', 'position': 'sticky', 'top': '0', 'zIndex': '1100',
            'backgroundColor': 'rgba(8,14,26,0.92)', 'backdropFilter': 'blur(12px)',
            'borderBottom': f"1px solid {C['border']}", 'display': 'flex',
            'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 20px',
        }),

        # BODY
        html.Div([

            # SIDEBAR
            html.Div(id='sidebar-panel', children=[
                html.Div([
                    html.Div('Sistema', style={
                        'fontSize': '10px', 'color': C['accent2'],
                        'textTransform': 'uppercase', 'letterSpacing': '2px',
                        'fontWeight': '700', 'marginBottom': '14px',
                    }),
                    status_dot('AWS IoT Core',  'aws-status'),
                    status_dot('MQTT Broker',   'mqtt-status'),
                    status_dot('Flux de dades', 'wifi-status'),
                ], style={**CARD, 'marginBottom': '12px'}),

                html.Div([
                    html.Div('Notes clíniques', style={
                        'fontSize': '10px', 'color': C['accent2'],
                        'textTransform': 'uppercase', 'letterSpacing': '2px',
                        'fontWeight': '700', 'marginBottom': '12px',
                    }),
                    dcc.Textarea(id='notes-area',
                        placeholder='Observacions, incidències o comentaris...',
                        style={
                            'width': '100%', 'height': '110px', 'borderRadius': '10px',
                            'border': f"1px solid {C['border']}", 'padding': '10px',
                            'resize': 'vertical', 'backgroundColor': C['surface'],
                            'color': C['text'], 'fontSize': '13px',
                            'fontFamily': FONT_SANS, 'boxSizing': 'border-box',
                        }),
                ], style={**CARD, 'marginBottom': '12px'}),
              html.Div([
                    html.Label('Nom de la sessió', style={
                        'fontSize': '11px', 'color': C['muted'],
                        'textTransform': 'uppercase', 'letterSpacing': '1px',
                        'marginBottom': '6px', 'display': 'block', 'fontWeight': '700',
                    }),
                    dcc.Input(
                        id='nom-sessio',
                        type='text',
                        placeholder='p.ex. Postura neutra matí...',
                        style=INPUT_STYLE
                    ),
                ], style={'marginBottom': '8px'}),              
                html.Div([
                    html.Button(id='btn-rec', n_clicks=0, children='⏺ Gravar sessió',
                        style={
                            **BTN,
                            'background': 'linear-gradient(135deg,#dc2626,#ef4444)',
                            'color': 'white', 'marginBottom': '8px',
                        }),
                    html.Button('↓ Exportar Excel', id='btn-xls', n_clicks=0, style={
                        **BTN,
                        'background': f'linear-gradient(135deg,#1d4ed8,{C["accent"]})',
                        'color': 'white',
                    }),
                    dcc.Download(id='down-xls'),
                    html.Button('⟳ Recalibrar baseline', id='btn-recal', n_clicks=0, style={
                        **BTN,
                        'background': f'linear-gradient(135deg,#065f46,{C["success"]})',
                        'color': 'white', 'marginTop': '8px',
                    }),
                    html.Div(id='rec-status', style={
                        'marginTop': '12px', 'padding': '10px', 'borderRadius': '10px',
                        'fontSize': '12px', 'fontFamily': FONT, 'textAlign': 'center',
                        'backgroundColor': C['surface'], 'color': C['muted'],
                        'border': f"1px solid {C['border']}",
                    }),
                  html.Button('⏸ Mode Repòs', id='btn-repos', n_clicks=0, style={
                        **BTN,
                        'background': f'linear-gradient(135deg,#1e3048,#334155)',
                        'color': C['muted'], 'marginTop': '8px',
                    }),
                    html.Button('👁 Mode Passiu', id='btn-passiu', n_clicks=0, style={
                        **BTN,
                        'background': f'linear-gradient(135deg,#78350f,{C["warning"]})',
                        'color': 'white', 'marginTop': '8px',
                    }),
                ], style=CARD),

            ], style={
                'width': '280px', 'minWidth': '280px', 'backgroundColor': C['sidebar'],
                'padding': '16px', 'borderRight': f"1px solid {C['border']}",
                'overflowY': 'auto', 'transition': 'all 0.25s ease',
                'display': 'flex', 'flexDirection': 'column',
            }),

            # CONTINGUT
            html.Div([
                # KPIs
                html.Div([
                    kpi_card('Pic màxim',       'kpi-max-val',   'kpi-max-sub',   C['danger']),
                    kpi_card('Pressió mitjana', 'kpi-avg-val',   'kpi-avg-sub',   C['accent']),
                    kpi_card('Estat del flux',  'kpi-state-val', 'kpi-state-sub', C['success']),
                    kpi_card('Mode dispositiu', 'kpi-mode-val', 'kpi-mode-sub', C['accent2']),

                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(180px,1fr))',
                    'gap': '12px', 'marginBottom': '16px',
                }),

                # HEATMAP TEMPS REAL
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div('Mapa de pressió · Temps real', style={
                                'fontSize': '15px', 'fontWeight': '700', 'color': C['text'],
                            }),
                            html.Div('Matriu 24×24 · Butterworth + EMA α=0.2 · γ=1.5', style={
                                'fontSize': '11px', 'color': C['muted'],
                                'marginTop': '3px', 'fontFamily': FONT,
                            }),
                        ]),
                        html.Div(id='last-update-label', style={
                            'fontSize': '12px', 'color': C['muted'], 'fontFamily': FONT,
                        }),
                    ], style={
                        'display': 'flex', 'justifyContent': 'space-between',
                        'alignItems': 'flex-start', 'marginBottom': '12px',
                        'flexWrap': 'wrap', 'gap': '8px',
                    }),
                    dcc.Graph(
                        id='grafic-coixi',
                        figure=build_empty_figure(),   # figura base — Patch actualitza z
                        config={'displaylogo': False, 'modeBarButtonsToRemove': [
                            'zoom2d','pan2d','select2d','lasso2d',
                            'zoomIn2d','zoomOut2d','autoScale2d','resetScale2d']},
                        style={'height': '65vh', 'minHeight': '480px'},
                    ),
                ], style={**CARD, 'marginBottom': '16px'}),

                # HISTORIAL
                html.Div([
                    html.Div([
                        html.Div('Historial de sessions', style={
                            'fontSize': '15px', 'fontWeight': '700', 'color': C['text'],
                        }),
                        html.Div(id='hist-info', style={
                            'fontSize': '11px', 'color': C['muted'],
                            'marginTop': '3px', 'fontFamily': FONT,
                        }),
                    ], style={'marginBottom': '14px'}),
                    dcc.Dropdown(
                        id='hist-dropdown',
                        placeholder='Selecciona una sessió...',
                        style={
                            'backgroundColor': C['surface'],
                            'color': C['text'],
                            'border': f"1px solid {C['border']}",
                            'borderRadius': '10px',
                            'marginBottom': '12px', } ),
                  
                    dcc.Slider(id='hist-slider', min=0, max=10, step=1, value=0,
                               marks=None,
                               tooltip={'placement': 'bottom', 'always_visible': True}),
                    dcc.Graph(id='grafic-historial',
                              config={'displaylogo': False},
                              style={'height': '55vh', 'minHeight': '420px',
                                     'marginTop': '16px'}),
                ], style=CARD),

            ], id='content-panel', style={
                'flex': '1', 'minWidth': '0', 'padding': '16px', 'overflowY': 'auto',
            }),

        ], style={'display': 'flex', 'height': 'calc(100vh - 64px)', 'overflow': 'hidden'}),

        dcc.Interval(id='int-100ms', interval=100,  n_intervals=0),
        dcc.Interval(id='int-1s',    interval=1000, n_intervals=0),
        dcc.Interval(id='int-3s',    interval=3000, n_intervals=0),
    ], style={'backgroundColor': C['bg'], 'minHeight': '100vh'})

# ==========================================
# 11. LAYOUT
# ==========================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=layout_login)
])

# ==========================================
# 12. CALLBACKS
# ==========================================

@app.callback(Output('live-clock', 'children'), Input('int-1s', 'n_intervals'))
def update_clock(n):
    return datetime.now().strftime('%d/%m/%Y  ·  %H:%M:%S')


@app.callback(
    [Output('page-content', 'children'), Output('login-msg', 'children')],
    Input('btn-login', 'n_clicks'),
    [State('user-in', 'value'), State('pass-in', 'value')]
)
def handle_login(n, u, p):
    global current_topic, mqtt_connected
    if not n:
        return no_update, no_update
    if not u or not p:
        return layout_login, html.Span('Introdueix usuari i contrasenya.',
                                        style={'color': C['danger']})
    try:
        res = taula_usuaris.get_item(Key={'Username': u})
        if 'Item' in res and res['Item']['Password'] == p:
            dev_id        = res['Item']['DeviceID']
            current_topic = f"coixi/{dev_id}/dades"
            try:
                mqtt_client.connect(ENDPOINT, 8883, 60)
                mqtt_client.subscribe(current_topic)
                mqtt_client.loop_start()
                mqtt_connected = True
                return layout_dashboard(u, dev_id), ''
            except Exception as e:
                mqtt_connected = False
                return layout_login, html.Span(f'Error MQTT: {str(e)[:80]}',
                                               style={'color': C['danger']})
        return layout_login, html.Span('Usuari o contrasenya incorrectes.',
                                        style={'color': C['danger']})
    except Exception as e:
        return layout_login, html.Span(f"Error AWS: {str(e)[:90]}",
                                        style={'color': C['danger']})


@app.callback(
    Output('store-ui', 'data'),
    Input('btn-sidebar', 'n_clicks'),
    State('store-ui', 'data'),
    prevent_initial_call=True
)
def toggle_sidebar(n, ui):
    ui = ui or {'sidebar_open': True}
    return {'sidebar_open': not ui.get('sidebar_open', True)}


@app.callback(
    [Output('sidebar-panel', 'style'), Output('content-panel', 'style')],
    Input('store-ui', 'data')
)
def update_sidebar_style(ui):
    is_open = (ui or {}).get('sidebar_open', True)
    sb = {
        'width':       '280px' if is_open else '0px',
        'minWidth':    '280px' if is_open else '0px',
        'backgroundColor': C['sidebar'],
        'padding':     '16px' if is_open else '0px',
        'borderRight': f"1px solid {C['border']}",
        'overflowY':   'auto' if is_open else 'hidden',
        'transition':  'all 0.25s ease',
        'display':     'flex', 'flexDirection': 'column',
    }
    cp = {'flex': '1', 'minWidth': '0', 'padding': '16px', 'overflowY': 'auto'}
    return sb, cp


@app.callback(
    [Output('store-rec', 'data'), Output('btn-rec', 'children'), Output('btn-rec', 'style')],
    Input('btn-rec', 'n_clicks'),
    [State('store-rec', 'data'), State('store-user', 'data')]  # ← afegeix store-user
)
def toggle_recording(n, rec, store):
    recording = (rec or {}).get('recording', False)
    if ctx.triggered_id == 'btn-rec':
        recording = not recording

    # Notifica l'ESP32 del canvi de mode via MQTT
    if store:
        dev_id = store.get('did', '')
        if recording:
            mqtt_client.publish(f"coixi/{dev_id}/control", "START_REC")
        else:
            mqtt_client.publish(f"coixi/{dev_id}/control", "STOP_REC")

    if recording:
        return ({'recording': True}, '⏹ Aturar gravació',
                {**BTN, 'background': 'linear-gradient(135deg,#7f1d1d,#dc2626)',
                 'color': 'white', 'marginBottom': '8px'})
    return ({'recording': False}, '⏺ Gravar sessió',
            {**BTN, 'background': 'linear-gradient(135deg,#dc2626,#ef4444)',
             'color': 'white', 'marginBottom': '8px'})

MODE_NOMS = {0: 'Repòs', 1: 'Passiu', 2: 'Actiu', -1: 'Desconnectat'}
MODE_COLORS = {0: C['muted'], 1: C['warning'], 2: C['success'], -1: C['danger']}

@app.callback(
    [
        Output('grafic-coixi',      'figure'),
        Output('rec-status',        'children'),
        Output('rec-status',        'style'),
        Output('kpi-max-val',       'children'),
        Output('kpi-max-sub',       'children'),
        Output('kpi-avg-val',       'children'),
        Output('kpi-avg-sub',       'children'),
        Output('kpi-state-val',     'children'),
        Output('kpi-state-sub',     'children'),
        Output('last-update-label', 'children'),
        Output('kpi-mode-val', 'children'),
        Output('kpi-mode-sub', 'children'),

    ],
    Input('int-100ms', 'n_intervals'),
    [State('notes-area', 'value'), State('store-user', 'data'), State('nom-sessio', 'value') ,State('store-rec', 'data')]
)
def refresh_data(n, notes, store, nom_sessio, rec_state):
    mode_txt   = MODE_NOMS.get(mode_global, 'Desconnectat')
    mode_color = MODE_COLORS.get(mode_global, C['danger']) 
    if not store:
        return no_update, '', {}, '—', '', '—', '', '—', '', '', '—', ''

    is_rec = (rec_state or {}).get('recording', False)

    # Gravació en thread separat — no bloqueja mai el render
    if is_rec:
        threading.Thread(
            target=guardar_a_dynamo_async,
            args=(store['uid'],
                  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  dades_globals.flatten().tolist(),
                  notes, nom_sessio),
            daemon=True
        ).start()
        rec_text   = '● Gravació activa'
        rec_color  = '#fecaca'
        rec_bg     = 'rgba(220,38,38,0.15)'
        rec_border = 'rgba(248,113,113,0.3)'
    else:
        rec_text   = 'Sistema preparat'
        rec_color  = '#6ee7b7'
        rec_bg     = 'rgba(34,197,94,0.1)'
        rec_border = 'rgba(52,211,153,0.25)'

    rec_style = {
        'marginTop': '12px', 'padding': '10px', 'borderRadius': '10px',
        'fontSize': '12px', 'fontFamily': FONT, 'textAlign': 'center',
        'backgroundColor': rec_bg, 'color': rec_color,
        'border': f'1px solid {rec_border}',
    }

    mat      = dades_globals
    max_val  = int(np.max(mat))
    avg_val  = float(np.mean(mat[mat > 50])) if np.any(mat > 50) else 0.0
    n_active = int(np.sum(mat > 50))

    # PATCH — envia només z i customdata al navegador, no reconstrueix la figura
    mat_norm = apply_heatmap_filters(mat)
    patched  = Patch()
    patched['data'][0]['z']          = mat_norm.tolist()
    patched['data'][0]['customdata'] = mat.astype(int).tolist()
    patched['layout']['annotations'] = [dict(
        xref='paper', yref='paper', x=0.0, y=1.04,
        text=f'<b>Màxim: {max_val} ADC</b>',
        showarrow=False, font=dict(size=13, color='#f87171')
    )]

    if last_mqtt_message:
        age       = int((datetime.now() - last_mqtt_message).total_seconds())
        last_txt  = f'fa {age}s'
        state_txt = 'Flux OK' if age <= 3 else 'Flux lent'
    else:
        last_txt  = 'Sense dades'
        state_txt = 'Offline'

    return (
        patched, rec_text, rec_style,
        f'{max_val}', 'ADC counts',
        f'{avg_val:.0f}', f'{n_active} cel·les actives',
        state_txt, last_txt,
        f'Última trama rebuda {last_txt}',
        mode_txt,        # ← kpi-mode-val
       'Mode ESP32',    # ← kpi-mode-sub
    )


@app.callback(
    [Output('kpi-state-val', 'style')],
    Input('int-3s', 'n_intervals')
)
def update_state_color(n):
    if last_mqtt_message:
        age = int((datetime.now() - last_mqtt_message).total_seconds())
        col = C['success'] if age <= 3 else C['warning']
    else:
        col = C['danger']
    return ({'fontSize': '20px', 'fontWeight': '900', 'color': col,
             'fontFamily': FONT, 'lineHeight': '1'},)


@app.callback(
    [Output('aws-status',      'children'), Output('mqtt-status',      'children'),
     Output('wifi-status',     'children'), Output('aws-status-dot',   'style'),
     Output('mqtt-status-dot', 'style'),    Output('wifi-status-dot',  'style')],
    Input('int-3s', 'n_intervals')
)
def update_status(n):
    dot_on   = {'width': '8px', 'height': '8px', 'borderRadius': '50%',
                'backgroundColor': C['success'], 'flexShrink': '0',
                'boxShadow': f'0 0 6px {C["success"]}'}
    dot_off  = {'width': '8px', 'height': '8px', 'borderRadius': '50%',
                'backgroundColor': C['danger'], 'flexShrink': '0'}
    dot_warn = {'width': '8px', 'height': '8px', 'borderRadius': '50%',
                'backgroundColor': C['warning'], 'flexShrink': '0'}
    if last_mqtt_message:
        age      = int((datetime.now() - last_mqtt_message).total_seconds())
        data_ok  = age <= 3
        data_txt = f'fa {age}s'
    else:
        data_ok  = False
        data_txt = 'sense dades'
    return (
        'Connectat',
        'Connectat' if mqtt_connected else 'Desconnectat',
        data_txt if data_ok else f'Atenció · {data_txt}',
        dot_on,
        dot_on if mqtt_connected else dot_off,
        dot_on if data_ok else dot_warn,
    )
  
def query_sessio_completa(uid, nom_sessio):
  items  = []
  kwargs = {
      'KeyConditionExpression': Key('PatientID').eq(uid),
      'FilterExpression':       Attr('NomSessio').eq(nom_sessio),
      'ScanIndexForward':       True,
  }
  while True:
      res    = taula_sessions.query(**kwargs)
      items += res.get('Items', [])
      if 'LastEvaluatedKey' not in res:
          break
      kwargs['ExclusiveStartKey'] = res['LastEvaluatedKey']
  return items

@app.callback(
    Output('down-xls', 'data'),
    Input('btn-xls', 'n_clicks'),
    [State('store-user', 'data'), State('hist-dropdown', 'value')],
    prevent_initial_call=True
)
def exportar_excel(n, store, sessio_seleccionada):
    if not store or not sessio_seleccionada:
        return no_update
    try:
        items = sorted(query_sessio_completa(store['uid'], sessio_seleccionada), key=lambda x: x['Timestamp'])      
        if not items:
            return no_update
        rows = []
        for item in items:
            m = json.loads(item['Matriu'])
            rows.append([
                item['Timestamp'],
                item.get('NomSessio', 'N/A'),
                item.get('Observacions', 'N/A'),
                item.get('MaxVal', '0'),
                item.get('AvgVal', '0'),
                item.get('NCeles', '0'),
            ] + m)
        cols = ['Data i Hora', 'NomSessio', 'Observacions',
                'Max ADC', 'Avg ADC', 'Cel·les actives'] + [f'Cel·la_{i}' for i in range(576)]
        df = pd.DataFrame(rows, columns=cols)
        return dcc.send_data_frame(df.to_excel,
                                   f"{sessio_seleccionada}.xlsx", index=False)
    except Exception as e:
        print(f"[Excel] Error: {e}")
        return no_update


# Callback 1 — carrega les sessions disponibles al dropdown
@app.callback(
    Output('hist-dropdown', 'options'),
    Input('store-user', 'data'),
    Input('int-3s', 'n_intervals')
)
def carregar_sessions(store, n):
    if not store:
        return []
    try:
        res = taula_sessions.query(
            KeyConditionExpression=Key('PatientID').eq(store['uid']),
            ScanIndexForward=False,
            Limit=200,
            ProjectionExpression='NomSessio, #ts',           # ← alias per Timestamp
            ExpressionAttributeNames={'#ts': 'Timestamp'}    # ← defineix l'alias
        )
        items = res.get('Items', [])
        sessions = {}
        for item in items:
            nom = item.get('NomSessio', 'Sessió sense nom')
            ts  = item.get('Timestamp', '')
            if nom not in sessions:
                sessions[nom] = ts
        return [
            {'label': f"{nom}  ·  {ts[:10]}", 'value': nom}
            for nom, ts in sessions.items()
        ]
    except Exception as e:
        print(f"[Dropdown] Error: {e}")   # ← ara veuràs l'error si n'hi ha
        return []

# Callback 2 — mostra les captures de la sessió seleccionada
@app.callback(
    [Output('hist-slider', 'max'), 
     Output('hist-info', 'children'),
     Output('grafic-historial', 'figure')],
    [Input('store-user', 'data'),
     Input('hist-dropdown', 'value'),
     Input('hist-slider', 'value')]
)
def update_historial(store, sessio_seleccionada, slider_val):
    empty = go.Figure()
    empty.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=C['bg'],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text='Selecciona una sessió', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14, color=C['muted']))]
    )
    if not store or not sessio_seleccionada:
        return 1, 'Selecciona una sessió per veure les captures.', empty
    try:
        items = query_sessio_completa(store['uid'], sessio_seleccionada)
        if not items:
            return 1, 'Cap captura en aquesta sessió.', empty
        items = sorted(items, key=lambda x: x['Timestamp'])
        maxim = len(items) - 1
        idx   = min(slider_val or 0, maxim)
        mat   = np.array(json.loads(items[idx]['Matriu'])).reshape(24, 24)
        fig   = build_heatmap_figure(mat)
        info  = f"Captura {idx+1}/{maxim+1}  ·  {items[idx]['Timestamp']}  ·  {items[idx].get('Observacions','N/A')}"

        temps   = [it['Timestamp'][11:19] for it in items]
        maxvals = [int(it.get('MaxVal', 0)) for it in items]
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temps, y=maxvals, mode='lines',
            line=dict(color='#f87171', width=2),
            name='Màxim ADC'
        ))
        fig_temp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(11,18,32,1)',
            margin=dict(l=40, r=20, t=30, b=40),
            title=dict(text='Evolució del valor màxim', font=dict(color='#94a3b8', size=12)),
            xaxis=dict(showgrid=False, color='#64748b', tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor='#1e3048', color='#64748b'),
            showlegend=False,
        )
        return maxim, info, fig
    except Exception as e:
        return 1, f'Error: {str(e)[:80]}', empty
      

@app.callback(
    Output('btn-recal', 'children'),
    Input('btn-recal', 'n_clicks'),
    State('store-user', 'data'),
    prevent_initial_call=True
)
def recalibrar(n, store):
    if store:
        dev_id = store.get('did', '')
        mqtt_client.publish(f"coixi/{dev_id}/control", "RECALIBRATE")
    return '⟳ Recalibrar baseline'
@app.callback(
    Output('btn-repos', 'children'),
    Input('btn-repos', 'n_clicks'),
    State('store-user', 'data'),
    prevent_initial_call=True
)
def forcar_repos(n, store):
    if store:
        mqtt_client.publish(f"coixi/{store.get('did','')}/control", "SET_REPOS")
    return '⏸ Mode Repòs'

@app.callback(
    Output('btn-passiu', 'children'),
    Input('btn-passiu', 'n_clicks'),
    State('store-user', 'data'),
    prevent_initial_call=True
)
def forcar_passiu(n, store):
    if store:
        mqtt_client.publish(f"coixi/{store.get('did','')}/control", "SET_PASSIU")
    return '👁 Mode Passiu'
if __name__ == '__main__':
    app.run(debug=False)
