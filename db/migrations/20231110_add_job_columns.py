'''Alembic migration script to add new columns to jobs table.'''

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20231110_add_job_columns"
down_revision = None  # set to previous revision if exists
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add new columns to the jobs table."""
    op.add_column(
        'jobs',
        sa.Column('dataset_id', sa.String(length=255), nullable=False)
    )
    op.add_column(
        'jobs',
        sa.Column('base_model', sa.String(length=255), nullable=True)
    )
    op.add_column(
        'jobs',
        sa.Column('priority', sa.Integer(), nullable=True, server_default='0')
    )
    op.add_column(
        'jobs',
        sa.Column('workflow_id', sa.String(length=255), nullable=True)
    )
    op.add_column(
        'jobs',
        sa.Column('artifacts', sa.JSON(), nullable=True)
    )
    op.add_column(
        'jobs',
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.func.now())
    )


def downgrade() -> None:
    """Remove the columns added in upgrade()."""
    op.drop_column('jobs', 'updated_at')
    op.drop_column('jobs', 'artifacts')
    op.drop_column('jobs', 'workflow_id')
    op.drop_column('jobs', 'priority')
    op.drop_column('jobs', 'base_model')
    op.drop_column('jobs', 'dataset_id')
