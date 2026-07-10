const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

// Demo data matching the inference module's default CLASS_NAMES
// (human-anomaly-detection-backend-main/inference.js) so /category and
// /item/:id return something real instead of empty arrays.
async function main() {
	const category = await prisma.category.upsert({
		where: { slug: 'human-behavior-anomalies' },
		update: {},
		create: { name: 'Human Behavior Anomalies', slug: 'human-behavior-anomalies' },
	});

	const item = await prisma.items.upsert({
		where: { slug: 'fall' },
		update: {},
		create: {
			name: 'Fall',
			slug: 'fall',
			category_id: category.id,
			description: 'A person falling or lying on the ground unexpectedly.',
		},
	});

	const fallClass = await prisma.classes.upsert({
		where: { class: 'Fall Detected' },
		update: {},
		create: { class: 'Fall Detected' },
	});
	const noFallClass = await prisma.classes.upsert({
		where: { class: 'No Fall' },
		update: {},
		create: { class: 'No Fall' },
	});

	await prisma.item_class_assign.upsert({
		where: { slug: 'fall-detected' },
		update: {},
		create: {
			name: 'Fall Detected',
			slug: 'fall-detected',
			item_id: item.id,
			class_id: fallClass.id,
		},
	});
	await prisma.item_class_assign.upsert({
		where: { slug: 'fall-not-detected' },
		update: {},
		create: {
			name: 'No Fall',
			slug: 'fall-not-detected',
			item_id: item.id,
			class_id: noFallClass.id,
		},
	});

	console.log('Seed complete.');
}

main()
	.catch((error) => {
		console.error(error);
		process.exitCode = 1;
	})
	.finally(() => prisma.$disconnect());
