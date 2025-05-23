#!/usr/bin/env python3
"""
Text-to-SQL Application
Converte consultas em linguagem natural para SQL e executa em bancos MySQL/PostgreSQL
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import pymysql
import psycopg2
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

class DatabaseConnector:
    """Gerencia conexões com bancos de dados MySQL e PostgreSQL"""
    
    def __init__(self):
        self.engine = None
        self.db_type = None
        self.connection_params = {}
    
    def connect_mysql(self, host: str, port: int, database: str, 
                     username: str, password: str) -> bool:
        """Conecta ao MySQL"""
        try:
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            self.db_type = "mysql"
            self.connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
            # Testa a conexão
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✓ Conectado ao MySQL: {database}@{host}:{port}")
            return True
        except Exception as e:
            print(f"✗ Erro ao conectar ao MySQL: {e}")
            return False
    
    def connect_postgresql(self, host: str, port: int, database: str, 
                          username: str, password: str) -> bool:
        """Conecta ao PostgreSQL"""
        try:
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            self.db_type = "postgresql"
            self.connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
            # Testa a conexão
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"✓ Conectado ao PostgreSQL: {database}@{host}:{port}")
            return True
        except Exception as e:
            print(f"✗ Erro ao conectar ao PostgreSQL: {e}")
            return False
    
    def get_tables_and_columns(self) -> Dict[str, List[Dict]]:
        """Obtém lista de tabelas e colunas do banco"""
        if not self.engine:
            return {}
        
        try:
            inspector = inspect(self.engine)
            schema_info = {}
            
            for table_name in inspector.get_table_names():
                columns = []
                for column in inspector.get_columns(table_name):
                    columns.append({
                        'name': column['name'],
                        'type': str(column['type']),
                        'nullable': column['nullable'],
                        'default': column.get('default')
                    })
                schema_info[table_name] = columns
            
            return schema_info
        except Exception as e:
            print(f"Erro ao obter schema: {e}")
            return {}
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Executa uma query SQL e retorna o resultado"""
        if not self.engine:
            raise Exception("Não conectado ao banco de dados")
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn)
            return result
        except Exception as e:
            raise Exception(f"Erro ao executar query: {e}")

class TextToSQLConverter:
    """Converte texto natural para SQL usando modelos de transformers"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Carrega modelo Text-to-SQL"""
        try:
            print("Carregando modelo Text-to-SQL...")
            # Usando T5 fine-tuned para Text-to-SQL
            model_name = "t5-small"  # Modelo leve para demonstração
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("✓ Modelo carregado com sucesso")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            print("Usando método de fallback baseado em regras")
    
    def convert_to_sql(self, natural_query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Converte consulta em linguagem natural para SQL"""
        if self.model and self.tokenizer:
            return self._convert_with_model(natural_query, schema_info)
        else:
            return self._convert_with_rules(natural_query, schema_info)
    
    def _convert_with_model(self, natural_query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Conversão usando modelo transformer"""
        try:
            # Prepara o contexto com informações do schema
            schema_context = self._build_schema_context(schema_info)
            prompt = f"Convert to SQL: {natural_query}\nSchema: {schema_context}"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
            sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return sql_query
        except Exception as e:
            print(f"Erro no modelo, usando fallback: {e}")
            return self._convert_with_rules(natural_query, schema_info)
    
    def _convert_with_rules(self, natural_query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Conversão usando regras básicas (fallback)"""
        query_lower = natural_query.lower()
        
        # Identifica tipo de operação
        if any(word in query_lower for word in ['média', 'average', 'avg']):
            return self._build_avg_query(natural_query, schema_info)
        elif any(word in query_lower for word in ['soma', 'total', 'sum']):
            return self._build_sum_query(natural_query, schema_info)
        elif any(word in query_lower for word in ['contar', 'count', 'quantos']):
            return self._build_count_query(natural_query, schema_info)
        elif any(word in query_lower for word in ['máximo', 'max', 'maior']):
            return self._build_max_query(natural_query, schema_info)
        elif any(word in query_lower for word in ['mínimo', 'min', 'menor']):
            return self._build_min_query(natural_query, schema_info)
        else:
            return self._build_select_query(natural_query, schema_info)
    
    def _build_schema_context(self, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói contexto do schema para o modelo"""
        context = []
        for table_name, columns in schema_info.items():
            col_names = [col['name'] for col in columns]
            context.append(f"{table_name}({', '.join(col_names)})")
        return "; ".join(context)
    
    def _build_avg_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query de média"""
        # Exemplo simples - busca por palavras-chave
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT AVG(*) FROM table_name;"
        
        table = tables[0]  # Simplificação
        numeric_cols = self._find_numeric_columns(schema_info[table])
        
        if numeric_cols:
            col = numeric_cols[0]
            return f"SELECT AVG({col}) as media FROM {table};"
        
        return f"SELECT AVG(*) FROM {table};"
    
    def _build_sum_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query de soma"""
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT SUM(*) FROM table_name;"
        
        table = tables[0]
        numeric_cols = self._find_numeric_columns(schema_info[table])
        
        if numeric_cols:
            col = numeric_cols[0]
            return f"SELECT SUM({col}) as total FROM {table};"
        
        return f"SELECT SUM(*) FROM {table};"
    
    def _build_count_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query de contagem"""
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT COUNT(*) FROM table_name;"
        
        table = tables[0]
        return f"SELECT COUNT(*) as total FROM {table};"
    
    def _build_max_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query de máximo"""
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT MAX(*) FROM table_name;"
        
        table = tables[0]
        numeric_cols = self._find_numeric_columns(schema_info[table])
        
        if numeric_cols:
            col = numeric_cols[0]
            return f"SELECT MAX({col}) as maximo FROM {table};"
        
        return f"SELECT MAX(*) FROM {table};"
    
    def _build_min_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query de mínimo"""
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT MIN(*) FROM table_name;"
        
        table = tables[0]
        numeric_cols = self._find_numeric_columns(schema_info[table])
        
        if numeric_cols:
            col = numeric_cols[0]
            return f"SELECT MIN({col}) as minimo FROM {table};"
        
        return f"SELECT MIN(*) FROM {table};"
    
    def _build_select_query(self, query: str, schema_info: Dict[str, List[Dict]]) -> str:
        """Constrói query SELECT básica"""
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT * FROM table_name;"
        
        table = tables[0]
        return f"SELECT * FROM {table} LIMIT 10;"
    
    def _find_numeric_columns(self, columns: List[Dict]) -> List[str]:
        """Encontra colunas numéricas"""
        numeric_types = ['int', 'integer', 'decimal', 'numeric', 'float', 'double', 'real']
        numeric_cols = []
        
        for col in columns:
            col_type = col['type'].lower()
            if any(num_type in col_type for num_type in numeric_types):
                numeric_cols.append(col['name'])
        
        return numeric_cols

class TextToSQLApp:
    """Aplicação principal Text-to-SQL"""
    
    def __init__(self):
        self.db_connector = DatabaseConnector()
        self.text_to_sql = TextToSQLConverter()
        self.schema_info = {}
    
    def show_menu(self):
        """Exibe menu principal"""
        print("\n" + "="*60)
        print("        TEXT-TO-SQL APPLICATION")
        print("="*60)
        print("1. Conectar ao MySQL")
        print("2. Conectar ao PostgreSQL")
        print("3. Visualizar Schema")
        print("4. Consulta em Linguagem Natural")
        print("5. Executar SQL Direto")
        print("6. Sair")
        print("="*60)
    
    def connect_database(self, db_type: str):
        """Interface para conectar ao banco"""
        print(f"\n--- Conectar ao {db_type.upper()} ---")
        
        try:
            host = input("Host (localhost): ").strip() or "localhost"
            port = input(f"Porta ({'3306' if db_type == 'mysql' else '5432'}): ").strip()
            port = int(port) if port else (3306 if db_type == 'mysql' else 5432)
            database = input("Nome do banco: ").strip()
            username = input("Usuário: ").strip()
            password = input("Senha: ").strip()
            
            if db_type == 'mysql':
                success = self.db_connector.connect_mysql(host, port, database, username, password)
            else:
                success = self.db_connector.connect_postgresql(host, port, database, username, password)
            
            if success:
                self.schema_info = self.db_connector.get_tables_and_columns()
                print(f"Schema carregado: {len(self.schema_info)} tabelas encontradas")
            
        except ValueError:
            print("Porta deve ser um número")
        except Exception as e:
            print(f"Erro: {e}")
    
    def show_schema(self):
        """Exibe informações do schema"""
        if not self.schema_info:
            print("Nenhum banco conectado")
            return
        
        print(f"\n--- SCHEMA DO BANCO ({self.db_connector.db_type.upper()}) ---")
        
        for table_name, columns in self.schema_info.items():
            print(f"\n📋 Tabela: {table_name}")
            print("-" * 40)
            for col in columns:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"  • {col['name']:<20} {col['type']:<15} {nullable}")
    
    def natural_language_query(self):
        """Interface para consulta em linguagem natural"""
        if not self.schema_info:
            print("Nenhum banco conectado")
            return
        
        print("\n--- CONSULTA EM LINGUAGEM NATURAL ---")
        print("Exemplos:")
        print("• Qual é a média de notas dos alunos?")
        print("• Quantos registros existem na tabela usuarios?")
        print("• Mostre o máximo valor da coluna preco")
        print()
        
        natural_query = input("Sua pergunta: ").strip()
        if not natural_query:
            return
        
        print("\n🔄 Convertendo para SQL...")
        try:
            sql_query = self.text_to_sql.convert_to_sql(natural_query, self.schema_info)
            print(f"\n📝 SQL Gerado:")
            print("-" * 40)
            print(sql_query)
            print("-" * 40)
            
            execute = input("\nExecutar query? (s/N): ").strip().lower()
            if execute in ['s', 'sim', 'y', 'yes']:
                self.execute_and_show_results(sql_query)
                
        except Exception as e:
            print(f"Erro na conversão: {e}")
    
    def execute_direct_sql(self):
        """Interface para executar SQL direto"""
        if not self.db_connector.engine:
            print("Nenhum banco conectado")
            return
        
        print("\n--- EXECUTAR SQL DIRETO ---")
        print("Digite sua query SQL (ou 'sair' para voltar):")
        
        sql_query = ""
        while True:
            line = input("SQL> ").strip()
            if line.lower() == 'sair':
                return
            if line.endswith(';'):
                sql_query += line
                break
            sql_query += line + " "
        
        if sql_query:
            self.execute_and_show_results(sql_query)
    
    def execute_and_show_results(self, sql_query: str):
        """Executa query e exibe resultados"""
        try:
            print("\n🔄 Executando query...")
            result_df = self.db_connector.execute_query(sql_query)
            
            print(f"\n📊 Resultados ({len(result_df)} registros):")
            print("=" * 60)
            
            if len(result_df) > 0:
                # Configura exibição do pandas
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 30)
                
                print(result_df.to_string(index=False))
                
                # Estatísticas básicas para colunas numéricas
                numeric_cols = result_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    print(f"\n📈 Estatísticas:")
                    print(result_df[numeric_cols].describe())
            else:
                print("Nenhum resultado encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao executar query: {e}")
    
    def run(self):
        """Executa aplicação principal"""
        print("Iniciando Text-to-SQL Application...")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEscolha uma opção: ").strip()
                
                if choice == '1':
                    self.connect_database('mysql')
                elif choice == '2':
                    self.connect_database('postgresql')
                elif choice == '3':
                    self.show_schema()
                elif choice == '4':
                    self.natural_language_query()
                elif choice == '5':
                    self.execute_direct_sql()
                elif choice == '6':
                    print("Encerrando aplicação...")
                    break
                else:
                    print("Opção inválida")
                    
                input("\nPressione Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\nEncerrando aplicação...")
                break
            except Exception as e:
                print(f"Erro: {e}")
                input("Pressione Enter para continuar...")

# Função para uso em Jupyter Notebook
def create_text_to_sql_interface():
    """Cria interface para Jupyter Notebook"""
    return TextToSQLApp()

# Execução como script standalone
if __name__ == "__main__":
    app = TextToSQLApp()
    app.run()