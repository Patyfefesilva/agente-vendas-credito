# agente-vendas-credito
import streamlit as st
import pandas as pd
import numpy as np
import groq
import os
from dotenv import load_dotenv

load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(page_title="Painel de Decisão Comercial & Crédito", layout="wide")

# --- INTERFACE STREAMLIT ---
st.title("📊 Agente Analítico de Vendas Inteligente & Crédito")
st.subheader("Poder de decisão em tempo real alimentado por arquivos de auditoria")

# --- VALIDAÇÃO E INICIALIZAÇÃO DA API KEY ---
# Tenta buscar das variáveis de ambiente do sistema (.env ou terminal)
api_key = os.environ.get("GROQ_API_KEY")

# Se não encontrar no sistema, cria um campo seguro na barra lateral do Streamlit
if not api_key:
    api_key = st.sidebar.text_input("🔑 Chave API do Groq (GROQ_API_KEY)", type="password")
    if not api_key:
        st.warning("⚠️ Chave API não encontrada! Por favor, insira sua GROQ_API_KEY na barra lateral para ativar o agente.")
        st.stop()  # Interrompe o script de forma limpa, evitando a tela vermelha de erro

# Inicialização segura do cliente Groq
try:
    client = groq.Groq(api_key=api_key)
except Exception as e:
    st.error(f"Erro crítico ao inicializar o cliente Groq: {str(e)}")
    st.stop()

# --- PROCESSAMENTO DOS DADOS REAIS ---
def processar_dados_producao(df):
    """
    Garante a presença das colunas obrigatórias e calcula o risco matemático real.
    """
    colunas_obrigatorias = [
        "Cliente", "Vendas_Atuais_Mês", "Projeção_Vendas_Futura", 
        "Score_Crédito_Interno", "Score_Mercado_Externo", "Limite_Atual", 
        "Inadimplência_Histórica", "Tendência_Setor_Externo"
    ]
    
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]
    if colunas_faltantes:
        st.error(f"O arquivo enviado está incompleto. Colunas ausentes: {colunas_faltantes}")
        return None

    df["Score_Crédito_Interno"] = pd.to_numeric(df["Score_Crédito_Interno"], errors='coerce').fillna(0)
    df["Score_Mercado_Externo"] = pd.to_numeric(df["Score_Mercado_Externo"], errors='coerce').fillna(0)
    
    df["Risco_Calculado"] = df.apply(
        lambda r: "Baixo" if (r["Score_Crédito_Interno"] + r["Score_Mercado_Externo"])/2 > 80 
        else ("Médio" if (r["Score_Crédito_Interno"] + r["Score_Mercado_Externo"])/2 > 60 else "Alto"), 
        axis=1
    )
    return df

# --- INTEGRAÇÃO COM AGENTE IA (GROQ) ---
def analisar_com_ia(dados_cliente, pergunta_usuario):
    prompt_contexto = f"""
    Você é um agente especialista em Vendas e Análise de Risco de Crédito para o setor de Comércio.
    Analise os seguintes dados REAIS do cliente para responder à dúvida do usuário ou fornecer a tomada de decisão comercial:
    
    - Cliente: {dados_cliente['Cliente']}
    - Vendas Atuais: R$ {dados_cliente['Vendas_Atuais_Mês']:,}
    - Projeção de Demanda Futura: R$ {dados_cliente['Projeção_Vendas_Futura']:,}
    - Score Interno: {dados_cliente['Score_Crédito_Interno']}/100
    - Score Externo (Mercado): {dados_cliente['Score_Mercado_Externo']}/100
    - Limite de Crédito Atual: R$ {dados_cliente['Limite_Atual']:,}
    - Histórico Interno de Inadimplência: {dados_cliente['Inadimplência_Histórica']}
    - Tendência do Setor do Cliente (Dados Externos): {dados_cliente['Tendência_Setor_Externo']}
    - Risco Combinado Calculado por Algoritmo: {dados_cliente['Risco_Calculado']}
    
    Se o usuário fizer uma pergunta genérica ou saudações, forneça uma análise estruturada profissional contendo:
    1. Possibilidade de aumento de vendas (Sim/Não e Justificativa baseada na projeção vs risco).
    2. Recomendação de alteração de limite (Se deve aumentar, manter ou reduzir, e o novo valor sugerido).
    3. Alerta de risco principal da operação.
    Responda sempre de forma direta, clara, sem preâmbulos e use tópicos.
    """

    conteudo_pergunta = pergunta_usuario if pergunta_usuario.strip() != "" else "Faça uma análise de risco completa e recomende a liberação de limite para este cliente."

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": prompt_contexto},
                {"role": "user", "content": conteudo_pergunta}
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erro ao consultar o agente Groq: {str(e)}"

# Componente de Upload para base de dados real
arquivo_upload = st.file_uploader(
    "Importe sua planilha de clientes (Formatos aceitos: CSV ou Excel)", 
    type=["csv", "xlsx"]
)

if arquivo_upload is not None:
    if arquivo_upload.name.endswith('.csv'):
        df_real = pd.read_csv(arquivo_upload)
    else:
        df_real = pd.read_excel(arquivo_upload)
        
    df_clientes = processar_dados_producao(df_real)
    
    if df_clientes is not None:
        # 1. Painel com a base de dados importada
        st.markdown("### 📋 Base de Clientes Carregada")
        st.dataframe(
            df_clientes, 
            column_config={
                "Vendas_Atuais_Mês": st.column_config.NumberColumn("Vendas Atuais", format="R$ %,.2f"),
                "Projeção_Vendas_Futura": st.column_config.NumberColumn("Proj. Futura", format="R$ %,.2f"),
                "Limite_Atual": st.column_config.NumberColumn("Limite Atual", format="R$ %,.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")

        # 2. Seletor dinâmico baseado na lista extraída do seu arquivo real
        st.markdown("### 🧠 Diagnóstico de Decisão por Cliente")
        cliente_selecionado = st.selectbox(
            "Selecione o cliente para gerar o parecer comercial e de limite:", 
            df_clientes["Cliente"].tolist()
        )

        if cliente_selecionado:
            dados_c = df_clientes[df_clientes["Cliente"] == cliente_selecionado].iloc[0]
            
            # Cards de resumo estatístico rápido do cliente real
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risco Calculado", dados_c["Risco_Calculado"])
                st.metric("Tendência de Setor", str(dados_c["Tendência_Setor_Externo"]))
            with col2:
                vendas_atuais = float(dados_c['Vendas_Atuais_Mês'])
                vendas_futuras = float(dados_c['Projeção_Vendas_Futura'])
                st.metric(
                    "Demanda Estimada", 
                    f"R$ {vendas_futuras:,.2f}", 
                    f"Variação: R$ {vendas_futuras - vendas_atuais:,.2f}"
                )
            with col3:
                score_medio = (float(dados_c['Score_Crédito_Interno']) + float(dados_c['Score_Mercado_Externo'])) / 2
                st.metric("Média de Score Crédito", f"{score_medio:.1f} / 100")
            
            st.markdown("---")
            
            # Campo de pergunta posicionado junto ao agente
            pergunta = st.text_input(
                f"💬 Digite uma pergunta específica para o agente sobre o cliente {cliente_selecionado}:",
                placeholder="Ex: Qual o principal risco se aumentarmos o limite dele em 30%?",
                key=f"input_{cliente_selecionado}"
            )
            
            with st.spinner(f"Agente Groq processando dados de {cliente_selecionado}..."):
                diagnostico_ia = analisar_com_ia(dados_c, pergunta)
                
            st.markdown("#### 🎯 Resposta e Parecer Técnico do Agente:")
            st.info(diagnostico_ia)
else:
    st.info("💡 Aguardando upload da planilha comercial/crédito para iniciar a análise.")
